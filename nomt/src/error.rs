use std::{any::Any, convert::Infallible, fmt::Debug};

#[derive(Debug)]
pub enum TaskError<E: std::error::Error> {
    Panic(Box<dyn Any + Send + 'static>, String),
    Error(E, String),
}

pub type InfallibleTaskResult<T> = TaskResult<T, Infallible>;

pub struct TaskResult<T, E: std::error::Error> {
    value: Option<T>,
    err: Option<TaskError<E>>,
    context: Option<String>,
}

impl<T, E: std::error::Error> From<std::thread::Result<Result<T, E>>> for TaskResult<T, E> {
    fn from(result: std::thread::Result<Result<T, E>>) -> Self {
        let mut value = None;
        let mut err = None;
        match result {
            Ok(Ok(val)) => {
                value.replace(val);
            }
            Ok(Err(inner_err)) => {
                err.replace(TaskError::Error(inner_err, "".to_string()));
            }
            Err(thread_err) => {
                err.replace(TaskError::Panic(thread_err, "".to_string()));
            }
        };
        Self {
            value,
            err,
            context: None,
        }
    }
}

//impl<T> From<std::thread::Result<std::io::Result<T>>> for TaskResult<T, std::io::Error> {
//fn from(result: std::thread::Result<std::io::Result<T>>) -> TaskResult<T, std::io::Error> {
//let mut value = None;
//let mut err = None;
//match result {
//Ok(Ok(val)) => {
//value.replace(val);
//}
//Ok(Err(io_err)) => {
//err.replace(TaskError::Error(io_err, "".to_string()));
//}
//Err(thread_err) => {
//err.replace(TaskError::Panic(thread_err, "".to_string()));
//}
//};
//Self {
//value,
//err,
//context: None,
//}
//}
//}

impl<T> From<std::thread::Result<T>> for TaskResult<T, Infallible> {
    fn from(result: std::thread::Result<T>) -> Self {
        let mut value = None;
        let mut err = None;
        match result {
            Ok(val) => {
                value.replace(val);
            }
            Err(thread_err) => {
                err.replace(TaskError::Panic(thread_err, "".to_string()));
            }
        };
        Self {
            value,
            err,
            context: None,
        }
    }
}

impl<T, E: std::error::Error> TaskResult<T, E> {
    pub fn with_context(&mut self, context: String) {
        self.context.replace(context);
    }
}

impl<T, E: std::error::Error> TaskResult<T, E> {
    pub fn handle(mut self) -> Result<T, E> {
        if let Some(value) = self.value {
            return Ok(value);
        }

        match self.err.take().unwrap() {
            TaskError::Panic(err_payload, context) => {
                eprintln!("{}", context);
                std::panic::resume_unwind(err_payload)
            }
            TaskError::Error(err, context) => {
                eprintln!("{}", context);
                Err(err)
                //let context = match io_err.kind() {
                //std::io::ErrorKind::Other => {
                //format!("{} - {}", context, io_err.get_ref().unwrap())
                //}
                //_ => context,
                //};
                //Err(std::io::Error::new(io_err.kind(), context))
            }
        }
    }
}

//impl<T> TaskResult<T, Infallible> {
//pub fn handle(mut self) -> T {
//if let Some(value) = self.value {
//return value;
//}
//
//match self.err.take().unwrap() {
//TaskError::Panic(err_payload, context) => {
//eprintln!("{}", context);
//std::panic::resume_unwind(err_payload)
//}
//TaskError::Error(..) => unreachable!(),
//}
//}
//}

//pub type TaskResult<T> = core::result::Result<T, TaskError>;
//

//pub fn from_spawned_thread<T>(
//res: std::thread::Result<std::io::Result<T>>,
//context: String,
//) -> TaskResult<T> {
//match res {
//Ok(Ok(val)) => Ok(val),
//Ok(Err(io_err)) => Err(TaskError::Io(io_err, context)),
//Err(thread_err) => Err(TaskError::Panic(thread_err, context)),
//}
//}
//
//pub fn handle_task_result<T>(task_res: TaskResult<T>) -> std::io::Result<T> {
//match task_res {
//Ok(res) => Ok(res),
//Err(TaskError::Panic(err_payload, context)) => {
//eprintln!("{}", context);
//std::panic::resume_unwind(err_payload)
//}
//Err(TaskError::Io(io_err, context)) => {
//let context = match io_err.kind() {
//std::io::ErrorKind::Other => format!("{} - {}", context, io_err.get_ref().unwrap()),
//_ => context,
//};
//Err(std::io::Error::new(io_err.kind(), context))
//}
//}
//}
