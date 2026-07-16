#ifndef RINLA_CORE_GMRFLIB_WRAPPER_H
#define RINLA_CORE_GMRFLIB_WRAPPER_H

typedef struct _IO_FILE FILE;
typedef struct GMRFLib_store_struct GMRFLib_store_tp;
typedef struct GMRFLib_design_opaque GMRFLib_design_tp;
typedef struct GMRFLib_tabulate_Qfunc_opaque GMRFLib_tabulate_Qfunc_tp;
typedef struct GMRFLib_idx_opaque GMRFLib_idx_tp;
typedef struct GMRFLib_idxval_opaque GMRFLib_idxval_tp;
typedef struct GMRFLib_idx2_opaque GMRFLib_idx2_tp;
typedef struct GMRFLib_idxsubmat_vector_opaque GMRFLib_idxsubmat_vector_tp;
typedef struct GMRFLib_str_opaque GMRFLib_str_tp;
typedef struct GMRFLib_constr_opaque GMRFLib_constr_tp;
typedef struct GMRFLib_matrix_opaque GMRFLib_matrix_tp;
typedef struct GMRFLib_density_opaque GMRFLib_density_tp;
typedef struct gsl_matrix gsl_matrix;
typedef struct gsl_vector gsl_vector;

#include "../../gmrflib/version.h"
#include "../../gmrflib/hash.h"
#include "../../gmrflib/hashP.h"
#include "../../gmrflib/GMRFLibP.h"
#include "../../gmrflib/graph.h"
#include "../../gmrflib/taucs.h"
#include "../../gmrflib/smtp-pardiso.h"
#include "../../gmrflib/sparse-interface.h"
#include "../../gmrflib/optimize.h"
#include "../../gmrflib/blockupdate.h"
#include "../../gmrflib/approx-inference.h"

#endif
