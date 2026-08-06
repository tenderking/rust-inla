use inla_math::CscMatrix;

pub trait LatentModel: Send + Sync {
    /// Generates the sparse precision matrix Q(theta) given hyperparameter values.
    fn build_precision(&self, theta: &[f64]) -> Result<CscMatrix, String>;

    /// Evaluates the mathematical log-prior density at the given hyperparameter values.
    fn log_prior_density(&self, theta: &[f64]) -> f64;

    /// Returns the expected number of hyperparameters (dimensionality of theta).
    fn num_hyperparameters(&self) -> usize;

    /// Optional hard linear constraints for this latent (default: none).
    fn constraints(&self) -> Option<&inla_math::ConstraintSpec> {
        None
    }
}

/// Generic closure wrapper enabling FFI / Python / R callbacks to implement LatentModel.
pub struct ClosureLatentModel<F1, F2>
where
    F1: Fn(&[f64]) -> Result<CscMatrix, String> + Send + Sync,
    F2: Fn(&[f64]) -> f64 + Send + Sync,
{
    build_precision_fn: F1,
    log_prior_density_fn: F2,
    num_hyper: usize,
}

impl<F1, F2> ClosureLatentModel<F1, F2>
where
    F1: Fn(&[f64]) -> Result<CscMatrix, String> + Send + Sync,
    F2: Fn(&[f64]) -> f64 + Send + Sync,
{
    pub fn new(build_precision_fn: F1, log_prior_density_fn: F2, num_hyper: usize) -> Self {
        Self {
            build_precision_fn,
            log_prior_density_fn,
            num_hyper,
        }
    }
}

impl<F1, F2> LatentModel for ClosureLatentModel<F1, F2>
where
    F1: Fn(&[f64]) -> Result<CscMatrix, String> + Send + Sync,
    F2: Fn(&[f64]) -> f64 + Send + Sync,
{
    fn build_precision(&self, theta: &[f64]) -> Result<CscMatrix, String> {
        (self.build_precision_fn)(theta)
    }

    fn log_prior_density(&self, theta: &[f64]) -> f64 {
        (self.log_prior_density_fn)(theta)
    }

    fn num_hyperparameters(&self) -> usize {
        self.num_hyper
    }
}

/// Type-erased closure wrapper enabling FFI / Python / R callbacks to implement `LatentModel`.
pub struct DynClosureLatentModel {
    build_precision_fn: Box<dyn Fn(&[f64]) -> Result<CscMatrix, String> + Send + Sync>,
    log_prior_density_fn: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    num_hyper: usize,
}

impl DynClosureLatentModel {
    pub fn new(
        build_precision_fn: Box<dyn Fn(&[f64]) -> Result<CscMatrix, String> + Send + Sync>,
        log_prior_density_fn: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
        num_hyper: usize,
    ) -> Self {
        Self {
            build_precision_fn,
            log_prior_density_fn,
            num_hyper,
        }
    }
}

impl LatentModel for DynClosureLatentModel {
    fn build_precision(&self, theta: &[f64]) -> Result<CscMatrix, String> {
        (self.build_precision_fn)(theta)
    }

    fn log_prior_density(&self, theta: &[f64]) -> f64 {
        (self.log_prior_density_fn)(theta)
    }

    fn num_hyperparameters(&self) -> usize {
        self.num_hyper
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inla_math::identity_csc;

    #[test]
    fn test_closure_latent_model() {
        let model = ClosureLatentModel::new(
            |theta| {
                let tau = theta[0].exp();
                identity_csc(5, tau)
            },
            |theta| -0.5 * theta[0] * theta[0],
            1,
        );

        assert_eq!(model.num_hyperparameters(), 1);
        assert_eq!(model.log_prior_density(&[2.0]), -2.0);
        let q = model.build_precision(&[0.0]).unwrap();
        assert_eq!(q.rows(), 5);
        assert_eq!(q.get(0, 0), Some(&1.0));
    }
}
