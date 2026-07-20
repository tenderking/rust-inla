//! Reusable scratch buffers for factorization / CCD (Symbolica-style pools).

use std::cell::RefCell;

#[cfg(feature = "sparse-ldlt")]
use std::sync::Arc;

#[cfg(feature = "sparse-ldlt")]
use faer::sparse::linalg::cholesky::SymbolicCholesky;

/// Cached symbolic LDLᵀ pattern for a fixed CSC sparsity (factorize-once / numeric-many).
#[cfg(feature = "sparse-ldlt")]
#[derive(Clone)]
pub struct SymbolicPatternCache {
    pub n: usize,
    /// CSC column pointers (`indptr`).
    pub indptr: Vec<usize>,
    /// CSC row indices.
    pub indices: Vec<usize>,
    pub symbolic: Arc<SymbolicCholesky<usize>>,
}

#[cfg(feature = "sparse-ldlt")]
impl std::fmt::Debug for SymbolicPatternCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymbolicPatternCache")
            .field("n", &self.n)
            .field("nnz", &self.indices.len())
            .finish()
    }
}

/// Scratch workspace reused across Newton / CCD evaluations on one thread.
#[derive(Debug, Default)]
pub struct LdltScratch {
    /// RHS / solve buffer (length `n`).
    pub rhs: Vec<f64>,
    /// Dense row-major workspace for the dense backend.
    pub dense: Vec<f64>,
    /// Unit-vector / diagonal-inverse column buffer.
    pub col: Vec<f64>,
    /// Sparse numeric factor values (faer `L` storage).
    pub l_values: Vec<f64>,
    /// Last symbolic pattern seen on this thread (AMD + etree).
    #[cfg(feature = "sparse-ldlt")]
    pub symbolic_cache: Option<SymbolicPatternCache>,
}

impl LdltScratch {
    pub fn with_capacity(n: usize, nnz_hint: usize) -> Self {
        Self {
            rhs: Vec::with_capacity(n),
            dense: Vec::with_capacity(n.saturating_mul(n)),
            col: Vec::with_capacity(n),
            l_values: Vec::with_capacity(nnz_hint),
            #[cfg(feature = "sparse-ldlt")]
            symbolic_cache: None,
        }
    }

    /// Ensure buffers can hold an `n × n` dense factorize + length-`n` solves.
    pub fn ensure_dense(&mut self, n: usize) {
        if self.dense.len() < n * n {
            self.dense.resize(n * n, 0.0);
        }
        if self.rhs.len() < n {
            self.rhs.resize(n, 0.0);
        }
        if self.col.len() < n {
            self.col.resize(n, 0.0);
        }
    }

    pub fn ensure_n(&mut self, n: usize) {
        if self.rhs.len() < n {
            self.rhs.resize(n, 0.0);
        }
        if self.col.len() < n {
            self.col.resize(n, 0.0);
        }
    }

    pub fn ensure_l_values(&mut self, len: usize) {
        if self.l_values.len() < len {
            self.l_values.resize(len, 0.0);
        }
    }

    /// True if `q` has the same CSC sparsity as the cached symbolic pattern.
    #[cfg(feature = "sparse-ldlt")]
    pub fn pattern_matches(&self, q: &crate::sparse::CscMatrix) -> bool {
        let Some(cache) = self.symbolic_cache.as_ref() else {
            return false;
        };
        if cache.n != q.rows() || q.rows() != q.cols() {
            return false;
        }
        let indptr_storage = q.indptr();
        let indptr = indptr_storage.raw_storage();
        let indices = q.indices();
        cache.indptr.as_slice() == indptr && cache.indices.as_slice() == indices
    }

    #[cfg(feature = "sparse-ldlt")]
    pub fn store_pattern(
        &mut self,
        q: &crate::sparse::CscMatrix,
        symbolic: Arc<SymbolicCholesky<usize>>,
    ) {
        let indptr_storage = q.indptr();
        self.symbolic_cache = Some(SymbolicPatternCache {
            n: q.rows(),
            indptr: indptr_storage.raw_storage().to_vec(),
            indices: q.indices().to_vec(),
            symbolic,
        });
    }
}

thread_local! {
    static THREAD_SCRATCH: RefCell<LdltScratch> = RefCell::new(LdltScratch::default());
}

/// Borrow the current thread's [`LdltScratch`] for the duration of `f`.
///
/// Rayon CCD workers each get their own TLS pool, avoiding cross-thread
/// allocation storms on Newton / diagonal-inverse hot paths.
pub fn with_thread_scratch<R>(f: impl FnOnce(&mut LdltScratch) -> R) -> R {
    THREAD_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        f(&mut scratch)
    })
}
