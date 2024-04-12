use crate::{
    backend::{Action, Db},
    timer::Timer,
    workload::Workload,
};

// All transfers happen between different accounts.
//
// + `size` refers to the amount of transfer performed by the workload
// + `percentage_cold_transfer` is the percentage of transfers to
//    a non-existing account, the remaining portion of transfers are to existing accounts.
//    It goes from 0 to 100
// + `additional_initial_capacity` is the amount of elements already present in the storage
// without counting all the accounts needed for the transfers.
#[derive(Debug, Clone)]
pub struct TransferWorkload {
    init_actions: Vec<Action>,
    run_actions: Vec<Action>,
}

impl TransferWorkload {
    pub fn new(size: u64, percentage_cold_transfer: u8, additional_initial_capacity: u64) -> Self {
        // `size` define the numer of transfer,
        // the total number of accounts used is `size * 2`
        //
        let n_sender_accounts = size;
        let sender_accounts = 0..n_sender_accounts;
        // (cold) non existing accounts
        let n_cold_accounts =
            (n_sender_accounts as f64 * (percentage_cold_transfer as f64 / 100.0)) as u64;
        // (warm) alredy existing accounts
        let n_warm_accounts = n_sender_accounts - n_cold_accounts;
        let warm_accounts = n_sender_accounts..n_sender_accounts + n_warm_accounts;
        // warm and cold accounts will receive balance from the senders

        let n_total_accouts = n_sender_accounts + n_cold_accounts + n_warm_accounts;

        // 0..n_sender_accounts are sender accounts
        // n_sender_accounts..n_sender_accounts + n_warm_accounts are warm receiver
        // n_sender_accounts + n_warm_accounts..n_total_accouts are cold receiver

        // prepare additional random entries
        let additional_keys = n_total_accouts..n_total_accouts + additional_initial_capacity;

        let init_balance = Some(1000u64.to_be_bytes().to_vec());
        let initial_writes: Vec<Action> = sender_accounts
            .clone()
            .chain(warm_accounts)
            .chain(additional_keys)
            .map(|id| Action::Write {
                key: id.to_be_bytes().to_vec(),
                value: init_balance.clone(),
            })
            .collect();

        let mut transfer_actions = vec![];
        let balance_from = Some(900u64.to_be_bytes().to_vec());
        let balance_warm_to = Some(1100u64.to_be_bytes().to_vec());
        let balance_cold_to = Some(100u64.to_be_bytes().to_vec());
        // create the transfer vector of actions
        for from in sender_accounts.into_iter() {
            let to = from + n_sender_accounts;
            let balance_to = if from < n_warm_accounts {
                balance_warm_to.clone()
            } else {
                balance_cold_to.clone()
            };

            let from = from.to_be_bytes().to_vec();
            let to = to.to_be_bytes().to_vec();

            transfer_actions.push(Action::Read { key: from.clone() });
            transfer_actions.push(Action::Read { key: to.clone() });
            transfer_actions.push(Action::Write {
                key: from,
                value: balance_from.clone(),
            });
            transfer_actions.push(Action::Write {
                key: to,
                value: balance_to,
            });
        }

        Self {
            init_actions: initial_writes,
            run_actions: transfer_actions,
        }
    }
}

impl Workload for TransferWorkload {
    fn init(&self, backend: &mut Box<dyn Db>) {
        backend.apply_actions(self.init_actions.clone(), None);
    }

    fn run(&self, backend: &mut Box<dyn Db>, timer: &mut Timer) {
        backend.apply_actions(self.run_actions.clone(), Some(timer));
    }
}
