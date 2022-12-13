#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "PrecisionTypes.h"

// https://github.com/olapiv/kth-applied-gpu-programming/blob/master/Final_Project/src/Particles.cu
/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}

/*__global__ void subcycle(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < param->ns)
    {
        /*printf("b\n");
        struct particles* part=NULL;
        printf("c\n");
        atomicExch(&(part->x[i]), part_ori->x[i]);
        printf("b\n");
        printf("%f %f \n", part->x[i], part_ori->x[i]);
        __syncthreads();
        atomicExch(&(part->y[i]), part_ori->y[i]);
        __syncthreads();
        atomicExch(&(part->z[i]), part_ori->z[i]);
        __syncthreads();
        atomicExch(&(part->u[i]), part_ori->u[i]);
        __syncthreads();
        atomicExch(&(part->v[i]), part_ori->v[i]);
        __syncthreads();
        atomicExch(&(part->w[i]), part_ori->w[i]);
        __syncthreads();
        printf("c\n");
        // auxiliary variables
        FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
        FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;

        // move each particle with new fields
        FPpart omdtsq, denom, ut, vt, wt, udotb;
        
        // local (to the particle) electric and magnetic field
        FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
        
        // interpolation densities
        int ix,iy,iz;//int * iarr = newArr1<int>(3); 
        FPfield weight[2][2][2]; //FPfield ***weight = newArr3<FPfield>(2, 2, 2);
        FPfield xi[2], eta[2], zeta[2];
        
        // intermediate particle position and velocity
        FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde; //, xtmp, ytmp, ztmp, utmp, vtmp, wtmp;
        
        // start subcycling
        for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++)
        {
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz]; //problem start
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }     
        }
        
        /*atomicExch(&(part_ori->x[i]), part->x[i]);
        __syncthreads();
        atomicExch(&(part_ori->y[i]), part->y[i]);
        __syncthreads();
        atomicExch(&(part_ori->z[i]), part->z[i]);
        __syncthreads();
        atomicExch(&(part_ori->u[i]), part->u[i]);
        __syncthreads();
        atomicExch(&(part_ori->v[i]), part->v[i]);
        __syncthreads();
        atomicExch(&(part_ori->w[i]), part->w[i]);
        __syncthreads();
        printf("ENd: %f %f \n", part->x[i], part_ori->x[i]);
    }
    
}*/
__global__ void subcycle(FPpart *x, FPpart *y, FPpart *z, FPpart *u, FPpart *v, FPpart *w, FPfield *Ex, FPfield *Ey, FPfield *Ez, FPfield *Bxn, FPfield *Byn, FPfield *Bzn, FPfield *XN, FPfield *YN, FPfield *ZN, struct parameters*param, struct particles* part, struct EMfield* field, struct grid* grd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%d, %d\n", i, part->nop);
    if (i < part->nop)
    {
        // auxiliary variables#
        FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
        FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;

        // move each particle with new fields
        FPpart omdtsq, denom, ut, vt, wt, udotb;
        
        // local (to the particle) electric and magnetic field
        FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
        
        // interpolation densities
        int ix,iy,iz;//int * iarr = newArr1<int>(3); 
        FPfield weight[2][2][2]; //FPfield ***weight = newArr3<FPfield>(2, 2, 2);
        FPfield xi[2], eta[2], zeta[2];
        
        // intermediate particle position and velocity
        FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde; //, xtmp, ytmp, ztmp, utmp, vtmp, wtmp;
        
        // start subcycling
        for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++)
        {
            xptilde = x[i];
            yptilde = y[i];
            zptilde = z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = x[i] - XN[(iz*(grd->nxn*grd->nyn)+(ix - 1)*grd->nyn+iy)]; //problem start
                eta[0]  = y[i] - YN[iz*(grd->nxn*grd->nyn)+(ix)*grd->nyn+(iy - 1)]; 
                zeta[0] = z[i] - ZN[(iz - 1)*(grd->nxn*grd->nyn)+(ix)*grd->nyn+(iy)];
                xi[1]   = XN[(iz)*(grd->nxn*grd->nyn)+(ix)*grd->nyn+(iy)] - x[i];
                eta[1]  = YN[(iz)*(grd->nxn*grd->nyn)+(ix)*grd->nyn+(iy)] - y[i];
                zeta[1] = ZN[(iz)*(grd->nxn*grd->nyn)+(ix)*grd->nyn+(iy)] - z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*Ex[(iz- kk)*(grd->nxn*grd->nyn)+(ix- ii)*grd->nyn+(iy -jj)]; 
                            Eyl += weight[ii][jj][kk]*Ey[(iz- kk)*(grd->nxn*grd->nyn)+(ix- ii)*grd->nyn+(iy -jj)];
                            Ezl += weight[ii][jj][kk]*Ez[(iz- kk)*(grd->nxn*grd->nyn)+(ix- ii)*grd->nyn+(iy -jj)];
                            Bxl += weight[ii][jj][kk]*Bxn[(iz- kk)*(grd->nxn*grd->nyn)+(ix- ii)*grd->nyn+(iy -jj)];
                            Byl += weight[ii][jj][kk]*Byn[(iz- kk)*(grd->nxn*grd->nyn)+(ix- ii)*grd->nyn+(iy -jj)];
                            Bzl += weight[ii][jj][kk]*Bzn[(iz- kk)*(grd->nxn*grd->nyn)+(ix- ii)*grd->nyn+(iy -jj)];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= u[i] + qomdt2*Exl;
                vt= v[i] + qomdt2*Eyl;
                wt= w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                x[i] = xptilde + uptilde*dto2;
                y[i] = yptilde + vptilde*dto2;
                z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            u[i]= 2.0*uptilde - u[i];
            v[i]= 2.0*vptilde - v[i];
            w[i]= 2.0*wptilde - w[i];
            x[i] = xptilde + uptilde*dt_sub_cycling;
            y[i] = yptilde + vptilde*dt_sub_cycling;
            z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    x[i] = x[i] - grd->Lx;
                } else { // REFLECTING BC
                    u[i] = -u[i];
                    x[i] = 2*grd->Lx - x[i];
                }
            }
                                                                        
            if (x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   x[i] = x[i] + grd->Lx;
                } else { // REFLECTING BC
                    u[i] = -u[i];
                    x[i] = -x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    y[i] = y[i] - grd->Ly;
                } else { // REFLECTING BC
                    v[i] = -v[i];
                    y[i] = 2*grd->Ly - y[i];
                }
            }
                                                                        
            if (y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    y[i] = y[i] + grd->Ly;
                } else { // REFLECTING BC
                    v[i] = -v[i];
                    y[i] = -y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    z[i] = z[i] - grd->Lz;
                } else { // REFLECTING BC
                    w[i] = -w[i];
                    z[i] = 2*grd->Lz - z[i];
                }
            }
                                                                        
            if (z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    z[i] = z[i] + grd->Lz;
                } else { // REFLECTING BC
                    w[i] = -w[i];
                    z[i] = -z[i];
                }
            }  

        }
    }
    if (i == 0)    
        printf("%f\n", z[i]); 
}

/** particle mover */
int mover_PC_gpu(struct particles* part_cpu, struct EMfield* field_cpu, struct grid* grd_cpu, struct parameters* param_cpu)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param_cpu->n_sub_cycles << " - species " << part_cpu->species_ID << " ***" << std::endl;
    //@@ Insert code below to allocate GPU memory here
    FPpart *x ;
    FPpart *y ;
    FPpart *z ;
    FPpart *u ;
    FPpart *v ;
    FPpart *w ;
    struct particles* part;

    struct EMfield* field;
    FPfield *Ex = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *Ey = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *Ez = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *Bxn = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *Byn = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *Bzn = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);

    FPfield *Ex_gpu = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *Ey_gpu = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *Ez_gpu = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *Bxn_gpu = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *Byn_gpu = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *Bzn_gpu = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);

    for (int i=0; i<grd_cpu->nxn; i++)
        for (int j=0; j<grd_cpu->nyn; j++)
            for (int k=0; k<grd_cpu->nzn; k++)
            {
                Ex[k*(grd_cpu->nxn*grd_cpu->nyn)+i*grd_cpu->nyn+j] = field_cpu->Ex[i][j][k];
                Ey[k*(grd_cpu->nxn*grd_cpu->nyn)+i*grd_cpu->nyn+j] = field_cpu->Ey[i][j][k];
                Ez[k*(grd_cpu->nxn*grd_cpu->nyn)+i*grd_cpu->nyn+j] = field_cpu->Ez[i][j][k];
                Bxn[k*(grd_cpu->nxn*grd_cpu->nyn)+i*grd_cpu->nyn+j] = field_cpu->Bxn[i][j][k];
                Byn[k*(grd_cpu->nxn*grd_cpu->nyn)+i*grd_cpu->nyn+j] = field_cpu->Byn[i][j][k];
                Bzn[k*(grd_cpu->nxn*grd_cpu->nyn)+i*grd_cpu->nyn+j] = field_cpu->Bzn[i][j][k];
            }
    
    struct grid* grd;
    FPfield *XN = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *YN = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *ZN = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);

    FPfield *XN_gpu = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *YN_gpu = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    FPfield *ZN_gpu = newArr1<FPfield>(grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn);
    for (int i=0; i<grd_cpu->nxn; i++)
        for (int j=0; j<grd_cpu->nyn; j++)
            for (int k=0; k<grd_cpu->nzn; k++)
            {
                XN[k*(grd_cpu->nxn*grd_cpu->nyn)+i*grd_cpu->nyn+j] = grd_cpu->XN[i][j][k];
                YN[k*(grd_cpu->nxn*grd_cpu->nyn)+i*grd_cpu->nyn+j] = grd_cpu->YN[i][j][k];
                ZN[k*(grd_cpu->nxn*grd_cpu->nyn)+i*grd_cpu->nyn+j] = grd_cpu->ZN[i][j][k];
            }

    
    struct parameters* param;
    cudaMalloc(&(part), (1)*sizeof(particles));
    cudaMalloc(&(x), (part_cpu->npmax)*sizeof(FPpart));
    cudaMalloc(&(y), (part_cpu->npmax)*sizeof(FPpart));
    cudaMalloc(&(z), (part_cpu->npmax)*sizeof(FPpart));
    cudaMalloc(&(u), (part_cpu->npmax)*sizeof(FPpart));
    cudaMalloc(&(v), (part_cpu->npmax)*sizeof(FPpart));
    cudaMalloc(&(w), (part_cpu->npmax)*sizeof(FPpart));
    cudaMalloc(&(field), (1)*sizeof(EMfield));
    cudaMalloc(&(Ex_gpu), (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield));
    cudaMalloc(&(Ey_gpu), (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield));
    cudaMalloc(&(Ez_gpu), (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield));
    cudaMalloc(&(Bxn_gpu), (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield));
    cudaMalloc(&(Byn_gpu), (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield));
    cudaMalloc(&(Bzn_gpu), (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield));
    cudaMalloc(&(grd), (1)*sizeof(grid));
    cudaMalloc(&(XN_gpu), (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield));
    cudaMalloc(&(YN_gpu), (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield));
    cudaMalloc(&(ZN_gpu), (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield));
    cudaMalloc(&param, (1)*sizeof(parameters));
    //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(part, part_cpu, (1)*sizeof(particles), cudaMemcpyHostToDevice);
    cudaMemcpy(x, part_cpu->x, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(y, part_cpu->y, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(z, part_cpu->z, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(u, part_cpu->u, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(v, part_cpu->v, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(w, part_cpu->w, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(field, field_cpu, (1)*sizeof(EMfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ex_gpu, Ex, (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ey_gpu, Ey, (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Ez_gpu, Ez, (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Bxn_gpu, Bxn, (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Byn_gpu, Byn, (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(Bzn_gpu, Bzn, (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd, grd_cpu, (1)*sizeof(grid), cudaMemcpyHostToDevice);
    cudaMemcpy(XN_gpu, XN, (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(YN_gpu, YN, (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(ZN_gpu, ZN, (grd_cpu->nxn*grd_cpu->nyn*grd_cpu->nzn)*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(param, param_cpu, (1)*sizeof(parameters), cudaMemcpyHostToDevice);
    //@@ Initialize the grid and block dimensions here
    int blockSize = 1024; // set this value
    int gridSize = (int)ceil((float)(part_cpu->nop)/blockSize);
    //@@ Launch the GPU Kernel here
    subcycle<<<gridSize, blockSize>>>(x, y, z, u, v, w, Ex_gpu, Ey_gpu, Ez_gpu, Bxn_gpu, Byn_gpu, Bzn_gpu, XN_gpu, YN_gpu, ZN_gpu, param, part, field, grd);
    cudaDeviceSynchronize();

    
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(part_cpu->x, x, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_cpu->y, y, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_cpu->z, z, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_cpu->u, u, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_cpu->v, v, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_cpu->w, w, (part_cpu->npmax)*sizeof(FPpart), cudaMemcpyDeviceToHost);

    //@@ Free the GPU memory here
    cudaFree(param); 
    cudaFree(part); 
    cudaFree(field); 
    cudaFree(grd);  
    cudaFree(x); 
    cudaFree(y); 
    cudaFree(z); 
    cudaFree(u); 
    cudaFree(v); 
    cudaFree(w);  
    delArr1(Ex);
    delArr1(Ey);
    delArr1(Ez);
    delArr1(Bxn);
    delArr1(Byn);
    delArr1(Bzn);
    delArr1(XN);
    delArr1(YN);
    delArr1(ZN);

    return(0); // exit succcesfully
} // end of the mover