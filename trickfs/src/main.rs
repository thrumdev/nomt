fn main() {
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();
    let handle = trickfs::spawn_trick("/tmp/trick").unwrap();
    println!("running...");
    std::io::stdin().read_line(&mut String::new()).unwrap();
    drop(handle);
}
