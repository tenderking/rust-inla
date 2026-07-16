use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rinla_core::{Eval1D, ar1_precision, ar1_precision_csc, laplace_newton_step};

fn bench_ar1_build(c: &mut Criterion) {
    let mut g = c.benchmark_group("ar1_build");
    for &n in &[100usize, 1_000, 5_000] {
        g.bench_with_input(BenchmarkId::new("dense_triplet", n), &n, |b, &n| {
            b.iter(|| ar1_precision(n, 0.7, 1.0).expect("ar1 triplet"))
        });
        g.bench_with_input(BenchmarkId::new("sprs_csc", n), &n, |b, &n| {
            b.iter(|| ar1_precision_csc(n, 0.7, 1.0).expect("ar1 csc"))
        });
    }
    g.finish();
}

fn bench_laplace_ldlt(c: &mut Criterion) {
    let mut g = c.benchmark_group("laplace_ldlt");
    for &n in &[100usize, 500, 1_000] {
        let q = ar1_precision_csc(n, 0.5, 1.0).expect("q");
        let evals = vec![
            Eval1D {
                logp: 0.0,
                grad: 0.1,
                hess: -1.0,
            };
            n
        ];
        g.bench_with_input(BenchmarkId::new("pure_rust_step", n), &n, |b, _| {
            b.iter(|| laplace_newton_step(&q, &evals).expect("step"))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_ar1_build, bench_laplace_ldlt);
criterion_main!(benches);
