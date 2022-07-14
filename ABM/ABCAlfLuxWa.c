
// ABCAlfLuxWa.c   - source code for ABCAlfLuxWa.DLL
//              Compiled under VC++ 2017


#include <string.h>
#include <stdio.h>
#include <io.h>
#include <process.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>


#include <ziggurat.h>
#include <mt19937ar.c>
#include <ziggurat.c>



#define IA 16807
#define IM 2147483647
#define AM (1.0/IM) 
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 2.23e-016
#define RNMX (1.0-EPS)
#define MASK 123459876 
#define pi 3.14159265358979323846 


__declspec (dllexport) int AlfLux1(double *xout, double *pout, double *pfout, double *params, double *Nxx, double *Nt, double *Ntrans, double *x0, double *p0, double *pf0, double *idm1,double *idm2);


__declspec (dllexport) int ALWd_con_mom(double *returns, double *moms_out, double *idm1, double *idm2,
	double *params, double *nx, double *nt, double *Ttrans, double *x0, double *Nemp, double *Nmoms);


__declspec (dllexport) int pdf_triv(double *ser1, double *ser2, double *ser3, double *com, double *dens);

__declspec (dllexport) int AlfLux_filt(double *xpred, double *ret, double *params, double *nx, double *nt, double *x0, double *Bsamp, double *idm1, double *idm2);


/*
** AlfLux1.dll    Compiled C++ code for model of Alfarano,Lux,Wagner, JEDC, 2008
**
** (C) Copyright T.L.
**
**  FORMAT    dllcall AlfLux1(xout,pout,pfout,params,Nx,Nt,x0,p0,pf0,init);
**
**
**  INPUT
**
**      xout = (Nt,1) vector for output of opinion index x
**      pout = (Nt,1) vector for output of log market price p
**      pfout = (Nt,1) vector for output of log fundamental price pf
**            
**      Note: Eq. (6) of Alfarano/Lux/Wagner has been used with instantaneous market clearing (beta -> inf)
**       
**      params    = (5,1) vector of parameters
**
**              params[1] :  parameter a
**				params[2]:   parameter b
**              params[3] :  parameter delta_t, i.e. micro time step, has to be < 1
**              params[4] :  parameter Nf*Tf
**              params[5] :  parameter Nc*Tc
**              params[6] :  parameter sig0: standard dev. of Brownian motion for log fundamental value
**  GLOBALS
**
**      Nx = scalar, number of agents
**      Nt = scalar, number of integer time periods to be recorded
**		Ntrans = scalar, number of transient periods
**      x0, p0, pf0 = scalars, starting values for x, p, pf
**      idm1,idm2 = scalars, value to initiate random number generator, has to be NEGATIVE INTEGER
*/



__declspec (dllexport) int AlfLux1(double *xout, double *pout, double *pfout, double *params,double *Nxx, double *Nt, double *Ntrans, double *x0, double *p0, double *pf0, double *idm1, double *idm2)

{
   	
	float Numsx,Numst,idu1,idu2,ntrans;
	
	int jj,Nx,Nomt,Nser,trobs;
			
	double a,b,NTf,NTc,sig0,omeg1,omeg2,x,r1,dW,pf,p,v1,v2,trun,tchange,lamda,jjcomp,diff;
	
	
	// Initialization of the Ziggurat random number generator
	uint32_t kn[128];
	float fn[128];
	float wn[128];

	uint32_t idum;

	uint32_t seed;

	idu1 = *idm1;
	idum = idu1;

	init_genrand(idum); 
	r4_nor_setup ( kn, fn, wn ); 
	
	idu2 = *idm2;
	seed= idu2;
	

	a = params[0];

	b = params[1];
	
	NTf = params[2];

	NTc = params[3];

	sig0 = params[4];
	
	Numsx = *Nxx;
		
	Nx = Numsx;

	
	Numst = *Nt;
		
	Nomt = Numst;

	ntrans = *Ntrans;
	
	trobs = ntrans;
	
	Nser = 0.5*((*x0)*Nx+Nx);  //no. of optimists at t = 0

	x = *x0; p = *p0; pf = *pf0; trun = 0.0; jj = 1;

	
// Loop through continuous time until Nt + Ntrans

	
	while (trun < *Nt + *Ntrans || jj <= Nomt + trobs) {

		//	Loop: Agents decide whether to switch

			jjcomp = jj;

			omeg1 = a + b*(*Nxx)*(1 + x)/2;	// transition rate to left

			omeg2 = a + b*(*Nxx)*(1 - x)/2;  // transition rate to right
			
			if (trun < jjcomp) {

			// The Gillespie algorithm
								
				v1 = genrand_real2();  // uniform random number
				
				lamda = 0.5*(1+x)*(*Nxx)*omeg2+0.5*(1-x)*(*Nxx)*omeg1;

				tchange = -log(v1)/lamda;

				trun += tchange;	// time for  next change
				
				v2 = genrand_real2(); // another random number decides which change takes place

				if (v2 <= 0.5*(1+x)*Nx*omeg2/lamda) Nser = Nser - 1;
	
				else Nser = Nser +1; 
	
				x = 2*Nser/(*Nxx) -1;

			}


			if (trun > jjcomp) {
			
				r1 = r4_nor(&seed,kn,fn,wn);
			
				dW = sig0*r1;		// if next integer time step is reeached: at fundamental innovation
				
				pf = pf*exp(dW);
			
				p = pf*exp((NTc/NTf)*x);  // price at integer time steps as function of fundamentals and sentiment
				
				if (jj > trobs) {	// if t > Ntrans: record output
				
					pfout[jj-trobs-1] = pf; 
					
					pout[jj-trobs-1] = p; 
					
					xout[jj-trobs-1] =x;
				}
				jj += 1;
			}
	}

	return 0;
}

/*
** Computation of moments differences between empirical series /returns) and simulation
**
** (C) Copyright T.L.
**
**  FORMAT    dllcall AlfLux_con_mom(returns,mom_out,idm1,idm2,params,nt,Ttrans,x0,Nemp,nx,Nmoms)
**
**
**  INPUT
**
**	returns: empirical series
**	mom_out: records moments differences
**	idm1, idm2: seeds of Ziggurat random number genereator, have to be negative integers	
**               
**      Note: Eq. (6) of Alfarano/Lux/Wagner has been used with instantaneous market clearing (beta -> inf)
**       
**      params    = (5,1) vector of parameters
**
**              params[1] :  parameter a
**				params[2]:   parameter b
**              params[3] :  parameter delta_t, i.e. micro time step, has to be < 1
**              params[4] :  parameter Nf*Tf
**              params[5] :  parameter Nc*Tc
**              params[6] :  parameter sig0: standard dev. of Brownian motion for log fundamental value
**  GLOBALS
**
**      Nx = scalar, number of agents
**      nt = scalar, number of integer time periods to be recorded
**	Ttrans = scalar, number of transient periods
**	Nemp: length of empirical series
**	Nmoms: number of moments (as listed in paper)
**      x0: scalar, starting value for x
*/


__declspec (dllexport) int ALWd_con_mom(double *returns, double *moms_out, double *idm1, double *idm2,
		double *params, double *nx, double *nt, double *Ttrans,double *x0, double *Nemp, double *Nmoms)
{
	float Numsx,Numst,idu,idu2,ntrans,nemp,Momno;
	
	int jj,i,Nx,Nomt,Nser,trobs,NN,Nmom;
	
	double a,b,NTf,NTc,sig0,omeg1,omeg2,x,xneu,xalt,r1,dW,pf,p,v1,v2,trun,tchange,lamda,jjcomp,diff;

	double *pout,*dp,*moms_sim,*moms_emp;
	
	// Initialization of the Ziggurat random number generator
	uint32_t kn[128];
	float fn[128];
	float wn[128];

	uint32_t idum;

	uint32_t seed;

	idu = *idm1;
	idum = idu;

	init_genrand(idum); 
	r4_nor_setup ( kn, fn, wn ); 
	
	idu2 = *idm2;
	seed= idu2;
	
	a = params[0];

	b = params[1];
	
	NTf = params[2];

	NTc = params[3];

	sig0 = params[4];

	//FILE *out;

	
	Numsx = *nx;
	Nx = Numsx;

	Numst = *nt;
	Nomt = Numst;

	ntrans = *Ttrans;
	trobs = ntrans;

	nemp = *Nemp;
	NN = nemp;

	Momno = *Nmoms;
	Nmom = Momno;

	//Memory allocation

	dp = (double *) malloc((Nomt-1)*sizeof(double));

	moms_sim = (double *) malloc(Nmom*sizeof(double));
	
	moms_emp = (double *) malloc(Nmom*sizeof(double));


	for (i=0;i<Nmom;i++) {

		moms_sim[i] = 0;

		moms_emp[i] = 0;

	}

	
	
	Nser = 0.5*((*x0)*Nx+Nx);	//no. of optimists at t = 0

	
// Loop through continuous time until Nt + Ntrans

	x = *x0; xneu = x; p = 1.0; pf = 1.0; trun = 0.0; jj = 1;
	
	
	while (trun < *nt + *Ttrans || jj <= Nomt + trobs) {

		
		//	Loop: Agents decide whether to switch

		jjcomp = jj;

		omeg1 = a + b * (*nx)*(1 + x) / 2;	// transition rate to left

		omeg2 = a + b * (*nx)*(1 - x) / 2;	// transition rate to right

		if (trun < jjcomp) {


			// The Gillespie algorithm

			v1 = genrand_real2();  // uniform random number

			lamda = 0.5*(1 + x)*(*nx)*omeg2 + 0.5*(1 - x)*(*nx)*omeg1;

			tchange = -log(v1) / lamda;

			trun += tchange;	// time for  next change
			
			v2 = genrand_real2(); // another random number decides which change takes place

			if (v2 <= 0.5*(1 + x)*(*nx)*omeg2 / lamda) Nser = Nser - 1;

			else Nser = Nser + 1;

			x = 2 * Nser / (*nx) - 1;


		}


		if (trun >= jjcomp) {

			r1 = r4_nor(&seed, kn, fn, wn);

			dW = sig0 * r1;	// if next integer time step is reeached: at fundamental innovation

			xalt = xneu;

			xneu = x;


			if (jj > trobs + 1) {		// after transient period

			/*
			out = fopen("output.dat","a");
			fprintf(out,"%f %f %d %f %f %f %f %f \n",trun,jjcomp,jj,x,trun-jjcomp,dW,pf,p);

			fclose(out);*/
										
				dp[jj - trobs - 2] = dW + (NTc / NTf)*(xneu - xalt); // return over unit time step

			}

			jj += 1;

		}

	}

	
	// Now the 15 moments are computes, first for the simulated series
	// First set of 4 moments
	
	for (i=0;i<Nomt-1;i++) {

		moms_sim[0] += dp[i]*dp[i];	// second moment

		moms_sim[3] += dp[i]*dp[i]*dp[i]*dp[i];	// fourth moment

	}



	for (i=1;i<Nomt-1;i++) {

		moms_sim[1] += dp[i]*dp[i-1];

		moms_sim[2] += dp[i]*dp[i]*dp[i-1]*dp[i-1];
		
	}

	// Moments 5 to 7

	if (Nmom >= 7) {

			for (i=1;i<Nomt-1;i++) {

				moms_sim[5] += fabs(dp[i])*fabs(dp[i-1]);
			}

			
			for (i=5;i<Nomt-1;i++) {

				moms_sim[4] += dp[i]*dp[i]*dp[i-5]*dp[i-5];

				moms_sim[6] += fabs(dp[i])*fabs(dp[i-5]);
			}
	}

	// Moments 8 to 11


	if (Nmom >= 11) {

			for (i=10;i<Nomt-1;i++) {

				moms_sim[7] += dp[i]*dp[i]*dp[i-10]*dp[i-10];

				moms_sim[8] += fabs(dp[i])*fabs(dp[i-10]);
			}

			
			for (i=15;i<Nomt-1;i++) {

				moms_sim[9] += dp[i]*dp[i]*dp[i-15]*dp[i-15];

				moms_sim[10] += fabs(dp[i])*fabs(dp[i-15]);
			}
	}

	// Moments 12 to 15
		
	if (Nmom == 15) {

			for (i=20;i<Nomt-1;i++) {

				moms_sim[11] += dp[i]*dp[i]*dp[i-20]*dp[i-20];

				moms_sim[12] += fabs(dp[i])*fabs(dp[i-20]);
			}

			
			for (i=25;i<Nomt-1;i++) {

				moms_sim[13] += dp[i]*dp[i]*dp[i-25]*dp[i-25];

				moms_sim[14] += fabs(dp[i])*fabs(dp[i-25]);
			}
	}

	// Normalize moments

	moms_sim[0] = moms_sim[0]/(Nomt-1); 
	
	moms_sim[3] = moms_sim[3]/(Nomt-1); 
	
	moms_sim[1] = moms_sim[1]/(Nomt-2);  
	
	moms_sim[2] = moms_sim[2]/(Nomt-2);

	if (Nmom >= 7) {

		moms_sim[4] = moms_sim[4]/(Nomt-6); moms_sim[5] = moms_sim[5]/(Nomt-2); moms_sim[6] = moms_sim[6]/(Nomt-6); 

	}


	if (Nmom >= 11) {

		moms_sim[7] = moms_sim[7]/(Nomt-11); moms_sim[8] = moms_sim[8]/(Nomt-11); moms_sim[9] = moms_sim[9]/(Nomt-16); moms_sim[10] = moms_sim[10]/(Nomt-16);

	}


	if (Nmom == 15) {

		moms_sim[11] = moms_sim[11]/(Nomt-21); moms_sim[12] = moms_sim[12]/(Nomt-21); moms_sim[13] = moms_sim[13]/(Nomt-26); moms_sim[14] = moms_sim[14]/(Nomt-26);

	}

	// Corresponding moments for empirical series
	
	for (i=0;i<NN;i++) {

		moms_emp[0] += (returns[i])*(returns[i]);

		moms_emp[3] += (returns[i])*(returns[i])*(returns[i])*(returns[i]);

	}

	for (i=1;i<NN;i++) {

		moms_emp[1] += (returns[i])*(returns[i-1]);

		moms_emp[2] += (returns[i])*(returns[i])*(returns[i-1])*(returns[i-1]);
		
	}

	
	if (Nmom >= 7) {

			for (i=1;i<NN;i++) {

				moms_emp[5] += fabs(returns[i])*fabs(returns[i-1]);
			}

			
			for (i=5;i<NN;i++) {

				moms_emp[4] += returns[i]*returns[i]*returns[i-5]*returns[i-5];

				moms_emp[6] += fabs(returns[i])*fabs(returns[i-5]);
			}
	}

	
	if (Nmom >= 11) {

			for (i=10;i<NN;i++) {

				moms_emp[7] += returns[i]*returns[i]*returns[i-10]*returns[i-10];
				
				moms_emp[8] += fabs(returns[i])*fabs(returns[i-10]);
			}

			
			for (i=15;i<NN;i++) {

				moms_emp[9] += returns[i]*returns[i]*returns[i-15]*returns[i-15];

				moms_emp[10] += fabs(returns[i])*fabs(returns[i-15]);
			}
	}


	if (Nmom == 15) {

			for (i=20;i<NN;i++) {

				moms_emp[11] += returns[i]*returns[i]*returns[i-20]*returns[i-20];
				
				moms_emp[12] += fabs(returns[i])*fabs(returns[i-20]);
			}

			
			for (i=25;i<NN;i++) {

				moms_emp[13] += returns[i]*returns[i]*returns[i-25]*returns[i-25];

				moms_emp[14] += fabs(returns[i])*fabs(returns[i-25]);
			}
	}

	moms_emp[0] = moms_emp[0]/NN; moms_emp[3] = moms_emp[3]/NN; moms_emp[1] = moms_emp[1]/(NN-1);  moms_emp[2] = moms_emp[2]/(NN-1);

	
	if (Nmom >= 7) {

		moms_emp[4] = moms_emp[4]/(NN-5); moms_emp[5] = moms_emp[5]/(NN-1); moms_emp[6] = moms_emp[6]/(NN-5); 

	}


	if (Nmom >= 11) {

		moms_emp[7] = moms_emp[7]/(NN-10); moms_emp[8] = moms_emp[8]/(NN-10); moms_emp[9] = moms_emp[9]/(NN-15); moms_emp[10] = moms_emp[10]/(NN-15);

	}


	if (Nmom == 15) {

		moms_emp[11] = moms_emp[11]/(NN-20); moms_emp[12] = moms_emp[12]/(NN-20); moms_emp[13] = moms_emp[13]/(NN-25); moms_emp[14] = moms_emp[14]/(NN-25);

	}

	// Finally, average moment differences are computed as output

	moms_out[0] =  moms_emp[0] - moms_sim[0]; moms_out[1] = moms_emp[1] - moms_sim[1]; moms_out[2] = moms_emp[2] - moms_sim[2];  moms_out[3] = moms_emp[3] - moms_sim[3];


	if (Nmom >= 7) {

		moms_out[4] =  moms_emp[4] - moms_sim[4]; moms_out[5] = moms_emp[5] - moms_sim[5]; moms_out[6] =  moms_emp[6] - moms_sim[6];  

	}


	if (Nmom >= 11) {

		moms_out[7] =  moms_emp[7] - moms_sim[7]; moms_out[8] = moms_emp[8] - moms_sim[8]; moms_out[9] =  moms_emp[9] - moms_sim[9]; moms_out[10] =  moms_emp[10] - moms_sim[10]; 

	}


	
	if (Nmom == 15) {

		moms_out[11] =  moms_emp[11] - moms_sim[11]; moms_out[12] = moms_emp[12] - moms_sim[12]; moms_out[13] =  moms_emp[13] - moms_sim[13]; moms_out[14] =  moms_emp[14] - moms_sim[14]; 

	}
	
	
	free(dp);

	return 0;
}



/*
** Computation of trivariate Normal density
**
** (C) Copyright T.L.
**
**  FORMAT    dllcall pdf_triv(double *ser1,double *ser2,double *ser3,double *com, double *dens)
**
**
**  INPUT
**
**	ser1, ser2, ser3: observations (scalars)mom_out: records moments differences
**	com: covariance matrix, as vector (stacked columnwise)
**               
**  OUTPUT
**
**      dens: trivariate Normal density
*/



__declspec (dllexport) int pdf_triv(double *ser1,double *ser2,double *ser3,double *com, double *dens)

{
   	double D,a11,a12,a13,a21,a22,a23,a31,a32,a33,A1,A2,A3,exps,AA;

	FILE *out;

	D = com[0]*(com[8]*com[4] - com[5]*com[7]) - com[1]*(com[8]*com[3] - com[5]*com[6]) + com[2]*(com[7]*com[3] - com[4]*com[6]);

	if (D <= 0) dens = 0;

	else {
	
		a11 =  (com[8]*com[4] - com[5]*com[7])/D;
	
		a21 = -(com[8]*com[1] - com[2]*com[7])/D; a12 = a21;
	
		a31 = (com[1]*com[5] - com[4]*com[2])/D; a13 = a31;
	
		a22 = (com[8]*com[0] - com[2]*com[6])/D;
	
		a23 = -(com[5]*com[0] - com[3]*com[2])/D;  a32 = a23;
	
		a33 = (com[0]*com[4] - com[1]*com[3])/D;

		A1 = a11*(*ser1) + a21*(*ser2) + a31*(*ser3);

		A2 = a12*(*ser1) + a22*(*ser2) + a32*(*ser3);

		A3 = a13*(*ser1) + a23*(*ser2) + a33*(*ser3);
	

		AA = -0.5*(A1*(*ser1) +A2*(*ser2) + A3*(*ser3)) - 1.5*log(2*pi) - 0.5*log(D);
	
		*dens = exp(AA);
	}

}

/*
** Computation of ABC filter for ALW model
**
** (C) Copyright T.L.
**
**  FORMAT    dllcall AlfLux_filt(xpred,ret,params,nx,nt,x0,Bsamp,idm1,idm2)
)
**
**
**  INPUT
**
**	ret: (pseudo-)empirical series of returns
**	idm1, idm2: seeds of Ziggurat random number genereator, have to be negative integers	
**  params    = (5,1) vector of parameters
**
**              params[1] :  parameter a
**				params[2]:   parameter b
**              params[3] :  parameter delta_t, i.e. micro time step, has to be < 1
**              params[4] :  parameter Nf*Tf
**              params[5] :  parameter Nc*Tc
**              params[6] :  parameter sig0: standard dev. of Brownian motion for log fundamental value
**  GLOBALS
**
**      nx = scalar, number of agents
**      nt = scalar, number of integer time periods to be recorded
**		x0 = vector, initial values of particles
**		Bsamp: number of particles
**
**  OUTPUT
**
**	xpred: vector: predcitions of sentiment
*/


__declspec (dllexport) int AlfLux_filt(double *xpred, double *ret, double *params, double *nx, double *nt, double *x0, double *Bsamp, double *idm1, double *idm2)

{
	
	// Initialization of the Ziggurat random number generator
	uint32_t kn[128];
	float fn[128];
	float wn[128];

	uint32_t idum;

	uint32_t seed;

	FILE *out;

	float Numsx, Numst, idu, idu2, bs;

	int Nx, Nomt, *Nser, samp, i, j, jj, indx, l;

	double a, b, sig, omeg1, omeg2, x, v1, v2, v3, trun, truni, tchange, lamda, *xout, *omeg, *mean_r, om_sum, t1, t2, NN, xtry, omtry, ii;

	idu = *idm1;
	idum = idu;

	init_genrand(idum); 
	r4_nor_setup(kn, fn, wn); 

	idu2 = *idm2;
	seed = idu2;


	a = params[0] / 1000;

	b = params[1] / 1000;

	sig = params[2] / 1000;

	Numsx = *nx;

	Nx = Numsx;


	Numst = *nt;

	Nomt = Numst;

	bs = *Bsamp;

	samp = bs;

	// Allocate memory to output vectors

	Nser = (int *)malloc(samp * sizeof(int));

	xout = (double *)malloc(samp * sizeof(double));

	omeg = (double *)malloc(samp * sizeof(double));

	mean_r = (double *)malloc(samp * sizeof(double));



// Loop through continuous time until Nt

	for (jj = 1; jj <= Nomt; jj++) {

		trun = jj - 1; om_sum = 0;

		for (i = 0; i < samp; i++) {		// Loop over particles

			truni = trun;

			x = x0[i];

			Nser[i] = 0.5*(x0[i] * Nx + Nx);

			while (truni < jj) {

				// The Gillespie algorithm

				omeg1 = a + b * (*nx)*(1 + x) / 2;

				omeg2 = a + b * (*nx)*(1 - x) / 2;

				v1 = genrand_real1();

				lamda = 0.5*(1 + x)*(*nx)*omeg2 + 0.5*(1 - x)*(*nx)*omeg1;

				tchange = -log(v1) / lamda;

				truni += tchange;

				v2 = genrand_real1();

				if (v2 <= 0.5*(1 + x)*Nx*omeg2 / lamda) Nser[i] = Nser[i] - 1;

				else Nser[i] = Nser[i] + 1;

				if (truni <= jj) x = 2 * Nser[i] / (*nx) - 1;
			}

			xout[i] = x;

			mean_r[i] = ret[jj - 1] - (xout[i] - x0[i]);  // difference between empirical return and return predicted by particle i

			omeg[i] = 1 / (mean_r[i] * mean_r[i]);		// resulting weights

			om_sum += omeg[i];

		}


		// Prediction: avarage forecast and normalization of weights

		omeg[0] = omeg[0] / om_sum;

		xpred[jj - 1] += xout[0] * omeg[0];

		for (i = 1; i < samp; i++) {

			xpred[jj - 1] += xout[i] * omeg[i] / om_sum;

			omeg[i] = omeg[i] / om_sum + omeg[i - 1];
		}

		// Binomial sampling

		for (i = 0; i < samp; i++) {

			v3 = genrand_real1();


			for (j = 0; j < samp; j++) {

				if (v3 <= omeg[j]) {

					x0[i] = xout[j];

					break;
				}
			}
		}


	}

	return 0;
}



/* Auxiliar function for transition from double/float to integer values */

int fround(double XX)

{
	int xnew;

   	if ( XX - floor(XX) < 0.5) xnew = floor(XX);

	else xnew = floor(XX) +1;
		
	return xnew;
}
