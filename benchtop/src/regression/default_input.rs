use crate::regression::{Isolate, RegressionInputs, Sequential, WorkloadInfo};

pub fn default_regression_input() -> RegressionInputs {
    RegressionInputs {
        workloads: [
            (
                "random_read_10".to_string(),
                WorkloadInfo {
                    name: "randr".to_string(),
                    size: 25_000,
                    initial_capacity: 10,
                    isolate: None,
                    sequential: Some(Sequential {
                        time_limit: None,
                        op_limit: Some(500_000),
                        mean: None,
                    }),
                },
            ),
            (
                "random_read_20".to_string(),
                WorkloadInfo {
                    name: "randr".to_string(),
                    size: 25_000,
                    initial_capacity: 20,
                    isolate: None,
                    sequential: Some(Sequential {
                        time_limit: None,
                        op_limit: Some(500_000),
                        mean: None,
                    }),
                },
            ),
            (
                "random_write_20".to_string(),
                WorkloadInfo {
                    name: "randw".to_string(),
                    size: 25_000,
                    initial_capacity: 20,
                    isolate: None,
                    sequential: Some(Sequential {
                        time_limit: None,
                        op_limit: Some(500_000),
                        mean: None,
                    }),
                },
            ),
            (
                "random_read_write_20".to_string(),
                WorkloadInfo {
                    name: "randrw".to_string(),
                    size: 25_000,
                    initial_capacity: 20,
                    isolate: None,
                    sequential: Some(Sequential {
                        time_limit: None,
                        op_limit: Some(500_000),
                        mean: None,
                    }),
                },
            ),
            (
                "transfer".to_string(),
                WorkloadInfo {
                    name: "transfer".to_string(),
                    size: 30_000,
                    initial_capacity: 20,
                    isolate: Some(Isolate {
                        iterations: 20,
                        mean: None,
                    }),
                    sequential: None,
                },
            ),
        ]
        .into(),
    }
}
