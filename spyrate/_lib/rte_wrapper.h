#ifndef _RTE_WRAPPER
#define _RTE_WRAPPER

#ifdef RTE_SINGLE_PRECISION
typedef float RTE_PRECISION;
#elif defined RTE_DOUBLE_PRECISION
typedef double RTE_PRECISION;
#endif


void apply_BC_gpt(int *ncol, int *nlay, int *ngpt,
        bool *top_at_1, RTE_PRECISION *inc_flux, RTE_PRECISION *fluxdn);

void apply_BC_factor(int *ncol, int *nlay, int *ngpt,
        bool *top_at_1, RTE_PRECISION *inc_flux,
        RTE_PRECISION *factor, RTE_PRECISION *fluxdn);

void apply_BC_0(int *ncol, int *nlay, int *ngpt,
        bool *top_at_1, RTE_PRECISION *fluxdn);

void lw_solver_noscat(int *ncol, int *nlay, int *ngpt,
        bool *top_at_1, RTE_PRECISION *D, RTE_PRECISION *weight,
        RTE_PRECISION *tau, RTE_PRECISION *lay_source,
        RTE_PRECISION *lev_source_inc, RTE_PRECISION *lev_source_dec, RTE_PRECISION *sfc_emis,
        RTE_PRECISION *sfc_src, RTE_PRECISION *radn_up, RTE_PRECISION *radn_down);

void lw_solver_noscat_GaussQuad(int *ncol, int *nlay, int *ngpt,
        bool *top_at_1, RTE_PRECISION *nmus,  RTE_PRECISION *Ds,
        RTE_PRECISION *weights, RTE_PRECISION *tau, RTE_PRECISION *lay_source,
        RTE_PRECISION *lev_source_inc, RTE_PRECISION *lev_source_dec, RTE_PRECISION *sfc_emis,
        RTE_PRECISION *sfc_src, RTE_PRECISION *flux_up, RTE_PRECISION *flux_down);

void lw_solver_2stream(int *ncol, int *nlay, int *ngpt,
        bool *top_at_1, RTE_PRECISION *tau, RTE_PRECISION *ssa,
        RTE_PRECISION *g, RTE_PRECISION *lay_source,
        RTE_PRECISION *lev_source_inc, RTE_PRECISION *lev_source_dec, RTE_PRECISION *sfc_emis,
        RTE_PRECISION *sfc_src, RTE_PRECISION *flux_up, RTE_PRECISION *flux_down);

void sw_solver_noscat(int *ncol, int *nlay, int *ngpt,
        bool *top_at_1, RTE_PRECISION *tau, RTE_PRECISION *mu0, RTE_PRECISION *flux_dir);

void sw_solver_2stream(int *ncol, int *nlay, int *ngpt,
        bool *top_at_1, RTE_PRECISION *tau, RTE_PRECISION *ssa,
        RTE_PRECISION *g, RTE_PRECISION *mu0,
        RTE_PRECISION *sfc_alb_dir, RTE_PRECISION *sfc_alb_dif, RTE_PRECISION *flux_up,
        RTE_PRECISION *flux_dn, RTE_PRECISION *flux_dir);

#endif
