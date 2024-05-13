use crate::{
    backend::Transaction,
    workload::{Init, Workload},
};

/// Create an initialization command for this.
pub fn init(num_accounts: u64) -> Init {
    Init {
        keys: (0..num_accounts).map(|id| encode_id(id).to_vec()).collect(),
        value: encode_balance(1000).to_vec(),
    }
}

fn encode_id(id: u64) -> [u8; 8] {
    id.to_be_bytes()
}

fn encode_balance(balance: u64) -> [u8; 8] {
    balance.to_be_bytes()
}

fn decode_balance(encoded: &[u8]) -> u64 {
    let mut buf = [0; 8];
    buf.copy_from_slice(encoded);
    u64::from_be_bytes(buf)
}

/// Build a new workload meant to emulate transfers.
///
/// `num_accounts` refers to the amount of accounts in the database.
///
/// `percentage_cold_transfer` ranges from 0 to 100 and indicates the proportion of transfers
/// which should be sent to a fresh account.
pub fn build(
    num_accounts: u64,
    workload_size: u64,
    percentage_cold_transfer: u8,
) -> TransferWorkload {
    TransferWorkload {
        num_accounts,
        workload_size,
        runs: 0,
        percentage_cold_transfer,
    }
}

/// A transfer-like workload.
pub struct TransferWorkload {
    /// The number of accounts in the system.
    pub num_accounts: u64,
    /// The size of the workload.
    pub workload_size: u64,
    /// The number of runs performed.
    pub runs: usize,
    /// The percentage of transfers to make to fresh accounts.
    pub percentage_cold_transfer: u8,
}

impl Workload for TransferWorkload {
    fn run(&mut self, transaction: &mut dyn Transaction) {
        let old_num_accounts = self.num_accounts as usize;

        let cold_sends =
            (old_num_accounts as f64 * (self.percentage_cold_transfer as f64 / 100.0)) as u64;
        let warm_sends = self.workload_size - cold_sends;

        let mut start_offset = (self.runs * self.workload_size as usize) % old_num_accounts;

        for i in 0..self.workload_size {
            // totally arbitrary choice.
            let send_account = start_offset as u64;
            let recv_account = if i < warm_sends {
                (old_num_accounts - start_offset) as u64
            } else {
                let a = self.num_accounts;
                self.num_accounts += 1;
                a
            };

            let send_balance = decode_balance(
                &transaction
                    .read(&encode_id(send_account))
                    .expect("account exists"),
            );
            let recv_balance = transaction
                .read(&encode_id(recv_account))
                .map_or(0, |v| decode_balance(&v));

            let new_send_balance = if send_balance == 0 {
                1000 // yay, free money.
            } else {
                send_balance - 1
            };
            let new_recv_balance = recv_balance + 1;

            transaction.write(
                &encode_id(send_account),
                Some(&encode_balance(new_send_balance)),
            );
            transaction.write(
                &encode_id(recv_account),
                Some(&encode_balance(new_recv_balance)),
            );

            start_offset = (start_offset + 1) % old_num_accounts;
        }

        self.runs += 1;
    }

    fn size(&self) -> usize {
        self.workload_size as usize
    }
}
