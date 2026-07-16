use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use inla_math::{CscMatrix, Eval1D, laplace_newton_step, sparse_from_triplets};

/// Local AR(1) CSC for math-only benches (avoids depending on inla_stats).
fn ar1_precision_csc(n: usize, rho: f64, tau: f64) -> Result<CscMatrix, String> {
    if n == 0 {
        return Err("n must be > 0".into());
    }
    let mut trips = Vec::with_capacity(3 * n);
    for i in 0..n {
        let diag = if i == 0 || i == n - 1 {
            tau
        } else {
            tau * (1.0 + rho * rho)
        };
        trips.push((i, i, diag));
        if i + 1 < n {
            trips.push((i, i + 1, -tau * rho));
            trips.push((i + 1, i, -tau * rho));
        }
    }
    Ok(sparse_from_triplets(n, n, &trips))
}

fn bench_ar1_build(c: &mut Criterion) {
    let mut g = c.benchmark_group("ar1_build");
    for &n in &[100usize, 1_000, 5_000] {
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
