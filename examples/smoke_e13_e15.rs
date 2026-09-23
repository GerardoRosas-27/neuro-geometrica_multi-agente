fn main() {
    let seed = 0xE1100u64;
    println!("running E13...");
    let r13 = cdt_rqm_epr::liquid_experiments_11_17::run_experiment_13(seed);
    println!("E13 verdict={} unseen={:.3} notes={}", r13.verdict, r13.accuracy_unseen, r13.notes);
    println!("running E15...");
    let r15 = cdt_rqm_epr::liquid_experiments_11_17::run_experiment_15(seed);
    println!("E15 verdict={} unseen={:.3} notes={}", r15.verdict, r15.accuracy_unseen, r15.notes);
}
