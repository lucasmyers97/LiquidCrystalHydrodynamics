c     --------------------------------------------------
c     --------------------------------------------------
c     --------------------------------------------------
c     ----------- BIHARMONIC SOLVER --------------------
c     --------------------------------------------------
c     --------------------------------------------------
c     --------------------------------------------------

      subroutine dbihar(g,a,b,c,d,bda,bdb,bdc,bdd,tolin,win,
     +                  iflagin,alpha,beta,m,n,idf,lw,
     +                  f,itcg,iflag,tol,w)
c
      integer m,n,idf,iflagin,iflag,itcg,lw
      double precision a,b,c,d,alpha,beta,tolin,tol
      double precision bda(n),bdb(n),bdc(m),bdd(m),f(m+2,n+2)
      double precision g(m+2,n+2),win(lw),w(lw)
c
c
c     this subroutine solves the equation
c
c     u    + 2*u    + u    + alpha*(u  +u  ) + beta*u = f(x,y)
c      xxxx     xxyy   yyyy         xx  yy
c
c
c     in the rectangle   a < x < b and c < y < d  ,
c
c     subject to boundary conditions of the first
c     kind, u and the exterior normal derivative
c     of u must be specified at the boundary.
c
c     the solution time is essentially proportional to the
c     number of gridpoints when the conjugate gradient
c     option is used.
c
c     this is version 2.0, april 1984.
c
c     this code is written by petter bjorstad, the author
c     acknowledges paul n swarztrauber for the use of his
c     fast fourier transform package and the linpack project
c     for the use of linear equation solvers.
c
c     this code and the mathematical theory behind it, is
c     described in detail in:
c
c          numerical solution of the biharmonic equation.
c
c          ph.d. dissertation, stanford university  1980.
c          copyright 1980, petter e. bjorstad.
c
c
c     any comments, questions or suggestions are most welcome
c     and should be directed to :
c
c                      petter e. bjorstad
c
c                      veritas research
c                      p.o. box 300
c                      n-1322 hovik
c                      norway
c
c     description of input variables.
c
c     a,b,c and d.    defines the rectangle.
c                     see above.
c
c     m               the number of interior grid points
c                     in the x-direction. the interval
c                     a.le.x.le.b is divided in (m+1) panels
c                     each of width (b-a)/(m+1).
c                     m must be odd and at least 3.
c                     the method is more efficient
c                     if m+1 is a product of small primes.
c
c     n               the number of interior grid points
c                     in the y-direction. the interval
c                     c.le.y.le.d is divided in (n+1) panels
c                     each of width (d-c)/(n+1).
c                     n must be odd and at least 3. the method is
c                     somewhat faster if n+1 is a product of small
c                     primes. it is also advisable to take n less
c                     than or equal to m.
c                     (for max speed and min storage.)
c
c     alpha           the constant alpha in the equation.
c
c     beta            the constant beta in the equation.
c
c     bda             array of dimension n containing the values
c                     of the exterior normal derivative of u on the
c                     side x=a ,y=c+j*(d-c)/(n+1)  ,j=1,..n .
c
c     bdb             array of dimension n containing the values
c                     of the exterior normal derivative of u on the
c                     side x=b ,y=c+j*(d-c)/(n+1)  ,j=1,..n .
c
c     bdc             array of dimension m containing the values
c                     of the exterior normal derivative of u on the
c                     side y=c ,x=a+i*(b-a)/(m+1)  ,i=1,..m .
c
c     bdd             array of dimension m containing the values
c                     of the exterior normal derivative of u on the
c                     side y=d ,x=a+i*(b-a)/(m+1)  ,i=1,..m .
c
c     f               array of dimension at least (m+2,n+2).
c                     f(i,j) must contain the values of the right
c                     hand side function f(x,y) evaluated at the
c                     points (x ,y ), where
c                              i  j
c                     x =a+(i-1)*(b-a)/(m+1) ,i=2,..m+1 .
c                      i
c                     y =c+(j-1)*(d-c)/(n+1) ,j=2,..n+1 .
c                      j
c
c                     f(1,j)   must contain the boundary values of u along
c                     the side x=a, y=y  ,j=2,..n+1 .
c                                      j
c                     f(m+2,j) must contain the boundary values of u along
c                     the side x=b, y=y  ,j=2,..n+1 .
c                                      j
c                     f(i,1)   must contain the boundary values of u along
c                     the side y=c, x=x  ,i=2,..m+1 .
c                                      i
c                     f(i,n+2) must contain the boundary values of u along
c                     the side y=d, x=x  ,i=2,..m+1 .
c                                      i
c
c     idf             rowdimension of array f as declared in
c                     the calling program.
c
c     tol             the conjugate gradient iteration is
c                     terminated using this parameter.
c                     the error in the solution of the discrete
c                     approximation (not to be confused with the
c                     truncation error) can be expected to be of the
c                     same order of magnitude provided that the
c                     function f has norm of magnitude unity.
c                     a good choice for tol is taking it a few
c                     orders of magnitude less than the expected
c                     truncation error.
c                     if the right hand side f in the given problem
c                     is many orders of magnitude smaller or bigger
c                     than one, then it is recommended that the user
c                     scales his problem.
c                     tol is a dummy variable if iflag is 3 or 4.
c
c     iflag
c              =1     this option is not available in version 2.0 .
c
c              =2     if this is the first solution on this grid
c                     using conjugate gradients with fourier trans-
c                     forms in both coordinate directions.
c                     there are no parameter restrictions, but
c                     the routine is only guaranteed to work when
c                     the discrete approximation is positive
c                     definite. this is always the case when alpha
c                     is nonpositive and beta is nonnegative.
c                     in other cases the user should monitor the
c                     output parameter itcg. if it stays larger
c                     than 15, the alternative iflag=4 is
c                     recommended.
c                     a call with iflag=2 restarts the low rank
c                     approximations used to speed up convergence.
c
c              =3     if this is the first solution on this grid
c                     using cholesky factorization.
c                     the parameters must satisfy alpha.le.0 ,
c                     beta.ge.0. (if this is violated the code
c                     will change iflag to 4.)
c
c              =4     if this is the first solution on this grid
c                     using an indefinite symmetric factorization.
c                     there are no parameter restrictions, but
c                     an error return will occur if the discrete
c                     system is computationally singular.
c
c     description of output variables.
c
c     f               contains the solution u of the discrete
c                     system in all gridpoints (x ,y ),i=1,..m+2 ,
c                                                i  j  j=1,..n+2 .
c
c     iflag
c
c     normal returns.  iflag will return with a value appropriate
c                      for repeated calls. it need normally not
c                      be changed by the user between calls to
c                      dbihar.
c                      however, if a sequence of problems is being
c                      solved where the parameters alpha or beta
c                      changes then itcg should be monitored and
c                      iflag should be reset to 2 when itcg
c                      increases (say above 15).
c                      also a change between the direct and
c                      iterative version must be set explicitly.
c
c                      if the code changes the value of iflag due
c                      to a change in the user input, a warning is
c                      printed indicating the reason for the change.
c
c                      iflag=2 new initial solution (see above).
c                      iflag=3 new initial solution (see above).
c                      iflag=4 new initial solution (see above).
c                      iflag=6 repeated solution after iflag=2.
c                      iflag=7 repeated solution after iflag=3.
c                      iflag=8 repeated solution after iflag=4.
c
c     error returns.   if iflag returns a negative value then
c                      an error was detected. the computed f(i,j)
c                      should be considered wrong. an error message
c                      giving the value of iflag is printed.
c
c              =-1     n and/or m was even or less than 3.
c              =-2     a.ge.b and/or c.ge.d .
c              =-3     idf.lt.m+2 or lw is too small.
c              =-4     linpack failure in cholesky-factorization.
c                      this should not occur,check input carefully.
c              =-5     linpack detected a computationally singular
c                      system using the symmetric indefinite
c                      factorization.
c              =-6     the conjugate gradient iteration failed to
c                      converge in 30 iterations. the probable
c                      cause is an indefinite or near singular
c                      system. try using iflag=4. note that tol
c                      returns an estimate of the residual in
c                      the current conjugate gradient iteration.
c
c     tol              an upper bound on the residuals obtained in
c                      the conjugate gradient iterations.
c                      tol will therefore normally be unchanged.
c
c     itcg             the number of conjugate gradient iterations.
c                      if this is large (say 20) then a direct
c                      solution using iflag =4 may be considered.
c
c     description of workspace.
c
c     lw              integer indicating the number of elements
c                     supplied in the workspace w.
c                     lw depends on iflag in the following way:
c                     (this can be improved slightly,version 3.0 ??)
c
c          iflag=  2: lw must be at least max(7*n,3*m)+2*(n+m)+19.
c                     if only one problem is solved on the grid then
c                     lw should be given its minimum value. (for
c                     maximum speed.)
c                     if several problems are to be solved then
c                     any larger lw will reduce the execution
c                     time for subsequent problems.(a low rank
c                     correction will be computed and used to
c                     improve the preconditioning that is used.)
c                     the code will not make use of lw larger than
c                     max(7*n,3*m)+2*(n+m)+19+20*(n+3) under any
c                     circumstance.
c
c         iflag=3,4:  lw must be at least max(3*m,4*n)+4*n+2*m
c                                        +0.5*(n+1)**2+19  .
c
c
c     w               a one dimensional array with at least
c                     lw elements.
c
c     subroutines and functions
c     included in this package.
c
c            1. biharmonic:    dbihar, dstart, dftrnx, dftrny,
c                              dbislf, dbisld, dpentf, dmatge,
c                              dtrigi, dhzeri, dhzero, dconju,
c                              dcmult, dpreco, dupdat.
c
c            2. fourier        dsinti, drffti, drfti1,
c               transform:     dsint,  drfftf, drftf1, dradf2,
c                              dradf3, dradf4, dradf5, dradfg.
c                              (by p.n. swarztrauber.)
c
c            3. linpack:       dspfa, dspsl, dppfa, dppsl.
c
c            4. blas:          idamax, daxpy, dcopy, ddot,
c                              dscal, dswap.
c
c            5. fortran:       mod,min,max,abs,
c                              atan,cos,sin,sqrt.
c
c     local.
c
c     biharmonic:         dstart, dftrnx, dftrny, dbislf, dbisld
c     fortran:            mod,min,max
c
c
      integer i1,i2,i3,i4,i5,i6,i7,i8
      integer nold,mold,iwf,iwl,il1,il2
      integer maxi
      double precision dx,dy,del,alf,bet
      double precision dxo,dyo,alfo,beto,wfo,wlo
c
c     copy input values into output to avoid inplace variables
c
      do i=1,m+2
          do j=1,n+2
              f(i,j) = g(i,j)
          enddo
      enddo
      tol = tolin
      iflag = iflagin
      do i=1,lw
          w(i) = win(i)
      enddo
c
c     check input for mistakes.
c
      il1  = max(7*n,3*m)+2*(n+m)
      il2  = max(4*n,3*m)+4*n+2*m+(n+1)**2/2+19
      if(n.lt.3.or.m.lt.3)               iflag = -1
      if(mod(m,2).eq.0.or.mod(n,2).eq.0) iflag = -1
      if(a.ge.b.or.c.ge.d)               iflag = -2
      if(idf.lt.m+2.or.lw.lt.il1)        iflag = -3
      if(iflag.lt.0)                     go to 1000
c
      dx   = (b-a)/(m+1.0d0)
      dy   = (d-c)/(n+1.0d0)
      del  = (dy/dx)**2
      alf  = alpha*dy*dy
      bet  = beta*dy**4
      maxi = (lw-il1)/(2*n+6)
      maxi = min(maxi,10)
c
      call dstart(m,n,alf,bet,f,idf,bda,bdb,bdc,bdd,dx,dy,del)
      call dftrnx(m,n,f(2,2),idf,w)
c
      if(iflag.eq.3.and.lw.lt.il2) go to 10
      if(iflag.le.4) go to 40
      if(n .ne.nold) go to 20
      if(m .ne.mold) go to 20
      if(dx.ne.dxo)  go to 20
      if(dy.ne.dyo)  go to 20
      if(w(iwf).ne.wfo.or.w(iwl).ne.wlo)  go to 30
      if(iflag.le.6) go to 50
      if(alfo.eq.alpha.and.beto.eq.beta) go to 50
      iflag= iflag-4
      write(6,2) iflag,alpha,beta
      go to 40
   10 iflag= 2
      write(6,5) iflag
      go to 40
   20 iflag= iflag-4
      write(6,1) iflag,n,m,a,b,c,d
      go to 40
   30 iflag= iflag-4
      write(6,3) iflag,iwf,iwl,wfo,wlo,w(iwf),w(iwl)
   40 nold = n
      mold = m
      dxo  = dx
      dyo  = dy
      alfo = alpha
      beto = beta
   50 go to (60,70,80,90,100,110,120,130),iflag
      go to 1000
   60 iwf  = max(4*n,3*m)+1
      iwl  = il1+(n+3)*maxi
      iflag= 2
      write(6,2) iflag,alpha,beta
   70 iwf  = max(4*n,3*m)+1
      iwl  = il1+(n+3)*maxi
      go to 200
   80 iwf  = max(3*m,4*n)+1
      iwl  = il2
      if(alpha.le.0.0d0.and.beta.ge.0.0d0) go to 200
      iflag= 4
      write(6,2) iflag,alpha,beta
   90 iwf  = max(3*m,4*n)+1
      iwl  = il2
      go to 200
  100 iflag= 2
      write(6,2) iflag,alpha,beta
  110 go to 200
  120 go to 200
  130 go to 200
  200 call dftrny(m,n,f(2,2),idf,w)
      if(iflag.eq.6) go to 250
      if(iflag.ne.2) go to 300
      i1   = 1
      i2   = i1+(n+1)/2
      i3   = i2+(n+1)/2
      i4   = i3+(n+1)/2
      i5   = i4+(n+1)/2
      i6   = max((5*m)/2,(7*n)/2)+17
      i7   = i6+2*(m+n)
      i8   = i7+2*maxi*(n+3)
  250 call dbislf(m,n,maxi,iflag,del,tol,alf,bet,itcg,idf,f(2,2),
     +     w(i1),w(i2),w(i3),w(i4),w(i5),w(i6),w(i7),w(i8))
      if(iflag.lt.0) go to 1000
      if(iflag.eq.2) iflag = 6
      go to 400
  300 if(iflag.eq.7.or.iflag.eq.8) go to 350
      i1   = 1
      i2   = i1+(n+1)/2
      i3   = i2+(n+1)/2
      i4   = max((5*m)/2,(7*n)/2)+17
      i5   = i4+2*(m+n)
  350 call dbisld(m,n,iflag,del,alf,bet,idf,f(2,2),
     +            w(i1),w(i2),w(i3),w(i4),w(i5))
      if(iflag.lt.0) go to 1000
      if(iflag.eq.3) iflag = 7
      if(iflag.eq.4) iflag = 8
  400 call dftrny(m,n,f(2,2),idf,w)
      call dftrnx(m,n,f(2,2),idf,w)
      wfo  = w(iwf)
      wlo  = w(iwl)
      return
 1000 write(6,4) iflag
    1 format(1x,'*warning*,iflag changed to ',i3,'n,m,maxi,a,b,c,d=',
     +       2i6,2x,4e12.2)
    2 format(1x,'*warning*,iflag changed to ',i3,'alpha,beta=',2e16.6)
    3 format(1x,'*warning*,iflag changed to ',i3,/1x,'element no ',
     +       i6,' and ',i6,' of w changed from ',2e16.6,' to ',
     +       2e16.6,' by user. ')
    4 format(/5x,'***error in dbihar, iflag= ',i6/)
    5 format(1x,'*warning*,iflag changed to ',i3,
     +       ' workspace needed, given : ',2i8)
      return
      end


      subroutine dbisld(m,n,iflag,del,alpha,beta,idf,f,
     +                  s,z,ws,trig,cmat)
c
      integer m,n,iflag,idf
      double precision del,alpha,beta
      double precision f(idf,*),cmat(*)
      double precision s(*),z(*),ws(*),trig(*)
c
c     direct solver for generalized biharmonic problem.
c     note that a real array is passed to linpack to store
c     pivot information if iflag is 4 or 8.
c     s,z and ws must have n2/2+1  elements each.
c     trig       must have 2*(m+n) elements.
c     cmat       must have (n+1)*(n+1)/2   if iflag is 3 or 7.
c     cmat       must have (n+1)*(n+1)/2+2*n if iflag is 4 or 8.
c
c     local.
c
c     biharmonic:         dtrigi, dpentf, dmatge
c     linpack:            dspfa,  dspsl,  dppfa, dppsl
c     blas:               dcopy,  daxpy,  dscal
c
      integer i,ki,kj,i1,i2
      integer m2,n2,ip,iq,info
      double precision scal1,scal2,x1,zero(1)
c
      zero(1) = 0.0d0
      if(iflag.eq.7.or.iflag.eq.8) go to 50
      x1     = 2.0d0/(n+1.0d0)
      scal1  = x1*(del/(m+1.0d0))**2
      scal2  = x1*.1250d0/(m+1.0d0)
      call dtrigi(m,del,trig,s)
      if(m.ne.n.or.del.ne.1.0d0) go to 40
      call dcopy(2*n,trig,1,trig(2*m+1),1)
      go to 50
   40 call dtrigi(n,1.0d0,trig(2*m+1),s)
   50 ip     = 1
      iq     = 0
      do 800 kj=1,2
        n2 = n/2+2-kj
        i2 = 2*m+1+(n+1)*(kj-1)
        if(iflag.eq.4.or.iflag.eq.8) iq=n2
        do 700 ki=1,2
          i1 = (m+1)*(ki-1)
          m2 = m/2+2-ki
          call dcopy(n2,zero,0,z,1)
          do 400 i = 1,m2
            call dcopy(n2,f(2*i+ki-2,kj),2*idf,s,1)
            x1 = scal1*trig(i1+i)
            call dpentf(n2,kj,trig(i1+m2+i),alpha,beta,trig(i2),s,s,ws)
            call daxpy(n2,x1,s,1,z,1)
            call dscal(n2,scal2,s,1)
            call dcopy(n2,s,1,f(2*i+ki-2,kj),2*idf)
  400       continue
c
c     the capacitance matrix equation is solved using linpack.
c
          if(iflag.eq.7) go to 450
          if(iflag.eq.8) go to 440
          call dmatge(m2,n2,ki,kj,del,alpha,beta,trig,cmat(ip+iq),ws)
          if(iflag.eq.3) go to 445
          call dspfa(cmat(ip+iq),n2,cmat(ip),info)
          if(info.ne.0) go to 1020
  440     call dspsl(cmat(ip+iq),n2,cmat(ip),z)
          go to 460
  445     call dppfa(cmat(ip),n2,info)
          if(info.ne.0) go to 1010
  450     call dppsl(cmat(ip),n2,z)
c
  460     do 600 i = 1,m2
            call dpentf(n2,kj,trig(i1+m2+i),
     +                 alpha,beta,trig(i2),z,s,ws)
            call daxpy(n2,-trig(i1+i),s,1,f(2*i+ki-2,kj),2*idf)
  600       continue
          ip = ip+iq+(n2*(n2+1))/2
  700     continue
  800   continue
      return
 1010 iflag  = -4
      return
 1020 iflag  = -5
      return
      end


      subroutine dmatge(m2,n2,ki,kj,del,alpha,beta,trig,cmat,w)
c
      integer m2,n2,ki,kj
      double precision del,alpha,beta
      double precision trig(*)
      double precision cmat(*)
      double precision w(*)
c
c     this routine computes the elements of the capacitance
c     matrix defined by ki and kj (ki=1 or 2, kj=1 or 2)
c     the matrix is stored in compact form in the array cmat.
c     this storage scheme is consistent with linpack.
c     operation count is m2*n2*(n2-1)/2 (mult+add)+4*m2*n2 (mult)
c                       +2*m2*n2 (div)+7*m2*n2 (add)  (plus order m2)
c
c     cmat must have at least n2*(n2+1)/2 elements. (this call.)
c     trig is assumed initialized by two calls to routine trigin.
c     workspace w must have at least n2 elements.
c
c     local.
c
c     blas:             ddot, daxpy
c
      integer i,j,k,km,ip,m,n,i1,i2
      double precision scal,x1,x2,x3,x4
      double precision ddot
c
      m    = 2*(m2+ki-2)+1
      n    = 2*(n2+kj-2)+1
      i1   = (m+1)*(ki-1)
      i2   = 2*m+(n+1)*(kj-1)
      x2   = 4.0d0/(n2+kj-1.0d0)
      scal = 4.0d0*del*del/(m2+ki-1.0d0)
c
c    the following loops (5 and 15) could be simplified
c    under fortran 77 assumptions. this is correct also
c    on fortran 66.
c
      ip   = 0
      do 15 k=1,n2
         if(k.eq.1) go to 10
         km = k-1
         do 5 j=1,km
            ip      = ip+1
            cmat(ip)= 0.0d0
    5       continue
   10    ip       = ip+1
         cmat(ip) = 1.0d0
   15    continue
      do 50 i=1,m2
         x1 = scal*trig(i1+i)*trig(i1+i)
         do 20 j=1,n2
            w(j) = trig(i2+j)/((trig(i1+m2+i)+trig(i2+n2+j))*
     +                         (trig(i1+m2+i)+trig(i2+n2+j)-alpha)+
     +                         beta)
   20       continue
         x3 = ddot(n2,trig(i2+1),1,w,1)
         ip = 0
         x3 = x1*x2/(1.0d0+x2*x3)
         do 40 k=1,n2
            x4 = -x3*w(k)
            call daxpy(k-1,x4,w,1,cmat(ip+1),1)
            ip = ip+k
            cmat(ip)=cmat(ip)+w(k)*(x1/trig(i2+k)+x4)
   40       continue
   50    continue
      return
      end


      subroutine dbislf(m,n,maxi,iflag,del,tol,alpha,beta,itcg,idf,f,
     +                  p,s,y,z,ws,trig,w3,diag)
c
      integer m,n,maxi,iflag,idf,itcg
      double precision del,tol,alpha,beta
      double precision f(idf,*)
      double precision p(*),s(*),y(*),z(*),diag(*)
      double precision trig(*),ws(*),w3(*)
c
c     conjugate gradient based solver of generalized
c     biharmonic equation.
c
c     p,s,y,z      must have at least (n/2+1) elements each.
c     ws           must have at least n+1 elements.
c     diag         must have at least 2*n elements.
c     w3           must have at least 2*max*(n+3) elements.
c
c     local.
c
c     biharmonic:         dtrigi, dhzeri, dpentf, dconju
c     blas:               dcopy,  dscal,  daxpy
c
      integer i,ki,kj,i1,i2,itc
      integer m2,n2,ip
      double precision scal1,scal2,x1,zero(1)
c
      zero(1)= 0.0d0
      itcg   = 0
      if(iflag.eq.6) go to 50
      x1     = 2.0d0/(n+1.0d0)
      scal1  = x1*(del/(m+1.0d0))**2
      scal2  = x1*.1250d0/(m+1.0d0)
      call dtrigi(m,del,trig,p)
      if(m.ne.n.or.del.ne.1.0d0) go to 30
      call dcopy(2*n,trig,1,trig(2*m+1),1)
      go to 40
   30 call dtrigi(n,1.0d0,trig(2*m+1),p)
   40 call dhzeri(m,n,1,del,alpha,beta,diag,trig,p)
c
   50 ip=1
      do 800 kj=1,2
        n2 = n/2+2-kj
        i2 = 2*m+1+(n+1)*(kj-1)
        do 700 ki=1,2
          m2 = m/2+2-ki
          i1 = (m+1)*(ki-1)
          call dcopy(n2,zero,0,z,1)
          call dcopy(n2,zero,0,y,1)
          do 400 i = 1,m2
            call dcopy(n2,f(2*i+ki-2,kj),2*idf,s,1)
            x1 = scal1*trig(i1+i)
            call dpentf(n2,kj,trig(i1+m2+i),alpha,beta,trig(i2),s,s,ws)
            call daxpy(n2,x1,s,1,z,1)
            call dscal(n2,scal2,s,1)
            call dcopy(n2,s,1,f(2*i+ki-2,kj),2*idf)
  400       continue
c
c     the capacitance matrix equation is solved
c     here using preconditioned conjugate gradients.
c
          call dconju(m2,n2,ki,kj,maxi,iflag,itc,del,tol,alpha,beta,
     +            z,s,p,y,trig,ws,diag(ip),w3)
c
c
          itcg = itcg+itc
          do 600 i = 1,m2
            call dpentf(n2,kj,trig(i1+m2+i),alpha,beta,trig(i2),y,s,ws)
            call daxpy(n2,-trig(i1+i),s,1,f(2*i+ki-2,kj),2*idf)
  600       continue
            ip=ip+n2
  700     continue
  800   continue
      itcg = itcg/4
      return
      end



      subroutine dconju(mm,nn,ki,kj,maxi,iflag,itcg,del,tol,alpha,
     +                  beta,z,s,p,y,trig,ws,diag,w3)
c
      integer mm,nn,ki,kj,maxi,iflag,itcg
      double precision del,tol,alpha,beta
      double precision z(*),s(*),p(*),y(*),trig(*),diag(*)
      double precision ws(*),w3(*)
c
c     conjugate gradient routine with preconditioning.
c     z,s,p,y and diag must have n elements.
c     (z,y and diag is input,y assumed zero.)
c
c     local.
c
c     biharmonic:         dpreco, dcmult, dupdat
c     blas:               idamax, ddot,   daxpy, dscal
c     fortran             abs,sqrt
c
      integer i1,i2,kr,it
      integer km(2,2)
      integer idamax
      double precision alf,aa,bet,bb,xx
      double precision ddot
c
      itcg = 0
      if(kj.eq.0) go to 5
      kr = kj
      i1 = maxi*(ki+2*kj-3)+1
      i2 = maxi*(ki+2*kj-3)*(nn+kj-1)+4*maxi+1
      go to 10
    5 kr = 1
      i1 = (ki-1)*maxi+1
      i2 = (ki-1)*maxi*nn+2*maxi+1
   10 if(iflag.le.2) km(ki,kr) = 0
      it = idamax(nn,z,1)
      if(abs(z(it)).lt.tol*tol) return
      itcg = 1
c
      call dpreco(nn,kj,iflag,maxi,km(ki,kr),
     +            p,z,diag,w3(i1),w3(i2),ws)
      alf  = ddot(nn,z,1,p,1)
      call dcmult(mm,nn,ki,kj,del,alpha,beta,p,s,trig,ws)
      aa   = alf/ddot(nn,p,1,s,1)
      call daxpy(nn,aa,p,1,y,1)
      call dupdat(nn,kj,maxi,iflag,km(ki,kr),tol,
     +            s,p,diag,w3(i1),w3(i2),ws)
c
      do 600 it=1,30
         call daxpy(nn,-aa,s,1,z,1)
         xx   = sqrt(ddot(nn,z,1,z,1))
         if(xx.lt.tol) return
         itcg = it+1
         bet  = alf
         call dpreco(nn,kj,iflag,maxi,km(ki,kr)-1,
     +               s,z,diag,w3(i1),w3(i2),ws)
         alf  = ddot(nn,z,1,s,1)
         bb   = alf/bet
         call dscal(nn,bb,p,1)
         call daxpy(nn,1.0d0,s,1,p,1)
         call dcmult(mm,nn,ki,kj,del,alpha,beta,p,s,trig,ws)
         aa   = alf/ddot(nn,p,1,s,1)
         call daxpy(nn,aa,p,1,y,1)
         call dupdat(nn,kj,maxi,iflag,km(ki,kr),tol,
     +              s,p,diag,w3(i1),w3(i2),ws)
  600    continue
      iflag=-6
      tol  = xx
      return
      end

      subroutine dupdat(nn,kj,maxu,iflag,km,tol,y,s,d,al,h,ws)
c
      integer nn,kj,maxu,iflag,km
      double precision tol
      double precision y(*),s(*)
      double precision d(*),al(*),h(nn,*),ws(*)
c
c     updates the current preconditioning matrix
c     using a symmetric rank one qn update. the
c     vectors are kept in h (never multiplied
c     together.
c     maxu is the maximum number of updates.
c     km is current number of updates.
c     y and s are nn vectors that defines a new update.
c     d is a diagonal matrix of dimension nn used in shzero.
c     h is array of size h(nn,maxu)
c     al is array of size al(maxu) used to keep scaling of the
c        symmetric rank one updates.
c     ws is workspace of lenght at least 3*nn+18 if kj=0.
c     if kj=1 or kj=2 then ws is a dummy argument.
c
c     local.
c
c     biharmonic:         dpreco
c     blas:               ddot, daxpy
c     fortran:            abs
c
      integer k
      double precision ddot
c
      if(km.eq.maxu) km = km+1
      if(km.eq.maxu+1) return
      k = km+1
      call dpreco(nn,kj,5,maxu,km,h(1,k),y,d,al,h,ws)
      call daxpy(nn,-1.0d0,s,1,h(1,k),1)
      al(k) = -ddot(nn,h(1,k),1,y,1)
      if(abs(al(k)).lt.tol*ddot(nn,h(1,k),1,h(1,k),1)) return
      al(k) = 1.0d0/al(k)
      km = km+1
      return
      end


      subroutine dpreco(nn,kj,iflag,maxu,km,p,z,d,al,h,ws)
c
      integer nn,kj,iflag,km,maxu
      double precision p(*),z(*)
      double precision d(*),al(*),h(nn,*)
      double precision ws(*)
c
c     preconditioning using a symmetric rank
c     one approximation to improve an initial
c     preconditioner defined in subroutine dhzero.
c     p=hz where h is the matrix defined above.
c     km rank one correctins are performed.
c     p and z are nn vectors.
c     d is diagonal of dimension nn used in shzero.
c     h is array of size h(nn,maxu)
c     al is array of size al(maxu) holding scaling factors
c     from the symmetric rank one update.
c     ws is workspace with at least 3*nn+18 elements if kj=0.
c     if kj=1 or kj=2 then ws is a dummy argument.
c
c     local.
c
c     biharmonic:         dhzero
c     blas:               ddot, daxpy
c     fortran:            min
c
      integer i,k
      double precision x1
      double precision ddot
c
      call dhzero(nn,kj,z,p,d,ws)
      if(iflag.le.2) return
      k = min(km,maxu)
      if(k.eq.0) return
      do 20 i = 1,k
         x1 = ddot(nn,h(1,i),1,z,1)*al(i)
         call daxpy(nn,x1,h(1,i),1,p,1)
   20    continue
      return
      end

      subroutine dhzero(nn,kj,x,y,d,ws)
c
      integer nn,kj
      double precision x(*),y(*),d(*),ws(*)
c
c     this routine preconditions the cg - iteration
c     by multipying with an approximation to the
c     inverse of the capacitance matrix.
c
c     x is input vector. x is not changed.
c     y is output vector. ie y=h0*x .
c     d is a diagonal matrix of dimension nn needed to define h0.
c     kj and ws are dummy parameters in this version of dhzero.
c
c     local.
c
      integer i
c
      do 10 i=1,nn
         y(i)=d(i)*x(i)
   10    continue
      return
      end

      subroutine dcmult(mm,nn,ki,kj,del,alpha,beta,x,y,trig,ws)
c
      integer mm,nn,ki,kj
      double precision del,alpha,beta
      double precision x(*),y(*)
      double precision trig(*),ws(*)
c
c     dcmult defines the capacitance-matrix.
c     y is c*x where c represents the matrix.
c
c     local.
c
c     biharmonic:         dpentf
c     blas:               dcopy,daxpy
c
      integer i,m,n,m2,n2,i1,i2
      double precision x1,scal
c
c     this is the case where two ffts are used.
c
      m2   = mm
      n2   = nn
      m    = 2*(m2+ki-2)+1
      n    = 2*(n2+kj-2)+1
      i1   = (m+1)*(ki-1)
      i2   = 2*m+1+(n+1)*(kj-1)
      scal = 4.0d0*del*del/(m2+ki-1.0d0)
      call dcopy(n2,x,1,y,1)
      do 30 i=1,m2
         x1 = scal*trig(i1+i)*trig(i1+i)
         call dpentf(n2,kj,trig(i1+m2+i),alpha,beta,
     +              trig(i2),x,ws,ws(n2+1))
         call daxpy(n2,x1,ws,1,y,1)
   30 continue
      return
      end

      subroutine dpentf(n2,kj,xl,alpha,beta,trig,y,x,ws)
c
      integer n2,kj
      double precision xl,alpha,beta
      double precision x(*),y(*)
      double precision trig(*),ws(*)
c
c     this routine helps solving special pentadiagonal systems
c     that arise when solving certain biharmonic problems.
c
c     y has length n2  (input )
c     x has length n2  (output)
c     trig is trigonometric information.
c     ws is workspace of lenght at least n2.
c
c     operation count is n2*(2(div)+4(mult)+6(add)).
c     n2(div) can be turned into n2(mult) by storing
c     1/trig(j) as well.
c
c     one mult and one add can be saved by precomputing the
c     quantity c1 (requiring slightly more storage.)
c     this will be implemented in later versions.
c
c     local.
c
c     blas:          ddot
c
      integer j
      double precision x1,c1,c2
      double precision ddot
c
c     form the inverse of the diagonal matrix.
c
      do 20 j=1,n2
         ws(j) = trig(j)/((xl+trig(n2+j))*(xl+trig(n2+j)-alpha)+beta)
   20   continue
      c1 = ddot(n2,trig,1,ws,1)
      c2 = ddot(n2,y,1,ws,1)
      x1 = 4.0d0/(n2+kj-1.0d0)
      c2 = x1*c2/(1.0d0+x1*c1)
      do 30 j=1,n2
         x(j) = (y(j)/trig(j)-c2)*ws(j)
   30    continue
      return
      end

      subroutine dhzeri(m,n,kj,del,alpha,beta,diag,trig,ws)
c
      integer m,n,kj
      double precision del,alpha,beta
      double precision diag(*),trig(*),ws(*)
c
c     this routine computes (initializes) the diagonal
c     matrix diag that is needed in routine dhzero.
c     diag has at least 2*n elements.
c     ws   has at least m/2+1 elements.
c
c     local.
c
      integer i,j,i1,i2,n2,m2,incr,kii,kjj,ip
      double precision x1,x2,x3
c
      x1 = .1250d0/(n+1.0d0)
      x2 = 8.0d0*del*del/(m+1.0d0)
      incr=1
      if(kj.eq.0) incr=2
      ip = 0
      do 800 kjj=1,2
         n2 = n/2+2-kjj
         i2 = 2*m+(n+1)*(kjj-1)
         do 700 kii = 1,2
            m2 = m/2+2-kii
            i1 = (m+1)*(kii-1)
            if(kj.eq.0) ip = (kii-1)*n+kjj-2
            do 50 i = 1,m2
               ws(i) = trig(i1+i)**2
   50          continue
            do 200 j = 1,n2
               x3 = 0.0d0
               ip = ip+incr
               do 100 i = 1,m2
                  x3 = x3+ws(i)/((trig(i1+m2+i)+trig(i2+n2+j))*
     +                         (trig(i1+m2+i)+trig(i2+n2+j)-
     +                          alpha)+beta)
  100             continue
               x3 = x2*x3+1.0d0
               diag(ip) = x1/x3
  200          continue
  700       continue
  800    continue
      return
      end

      subroutine dtrigi(n,del,trig,w)
c
      integer n
      double precision del
      double precision trig(*),w(*)
c
c     this routine computes trigonometric information needed to
c     solve the biharmonic equation.
c
c     input.
c            n   is assumed to be odd and at least 3.
c            del is (dy/dx)**2.
c     output.
c            trig(i), i=1,2.. 2*n.
c
c     workspace.
c            w  must have at least n/2+(n/2+1)/2 elements.
c
c            sin(i*pi/(n+1)) and 2*del*(1-cos(i*pi/(n+1)) are
c            computed for i=1,2,..n and stored in trig in the
c            following way.
c
c            odd  sin   i=1,3,5,...
c            odd  cos   i=1,3,5,...
c            even sin   i=2,4,6,...
c            even cos   i=2,4,6,...
c
c     local.
c
c     fortran:        atan,sin
c
      integer n2,n4,i
      double precision pi,ar,del2,del4
c
      pi  = 4.0d0*atan(1.0d0)
      ar  = pi/(n+1.0d0)
      del2= 2.0d0*del
      del4= 2.0d0*del2
      n2  = n/2
      n4  = (n2+1)/2
      do 10 i=1,n2
         w(i)=sin(i*ar)
   10    continue
      ar  = .50d0*ar
      do 20 i=1,n4
         w(n2+i)=del4*sin((2*i-1)*ar)**2
   20    continue
      trig(n4+1)    = 1.0d0
      trig(n2+n4+2) = del2
      do 30 i=1,n4
         trig(i)      = w(2*i-1)
         trig(n2+2-i) = w(2*i-1)
         trig(n2+1+i) = w(n2+i)
         trig(n+2-i)  = del4-w(n2+i)
   30    continue
      trig(n+n4+1)   = 1.0d0
      trig(n+n2+n4+1)= del2
      n4  = n2/2
      if(n4.eq.0) return
      do 40 i=1,n4
         trig(n+1+i)    = w(2*i)
         trig(n+n2+2-i) = w(2*i)
         trig(n+n2+1+i) = del4*w(i)**2
         trig(2*n+1-i)  = del4-trig(n+n2+1+i)
   40    continue
      return
      end


      subroutine dftrny(m,n,f,idf,ws)
c
      integer m,n,idf
      double precision f(idf,*)
      double precision ws(*)
c
c     performs m sine transforms. the transform is unscaled.
c     note that this transform must be scaled up by a factor
c     two in order to correspond to the transform described
c     in chapter 4.2 of numerical solution of the biharmonic
c     equation.
c     n should be odd. (when dsint is called.)
c     workspace ws must have at least int(3.5*n+16) elements.
c
c     local.
c
c     fourier:            dsinti, dsint   (swarztrauber version 3.)
c     blas:               dcopy
c
      integer i
c
      call dsinti(n,ws(n+2))
      do 500 i   =1,m
         call dcopy(n,f(i,1),idf,ws,1)
         call dsint(n,ws,ws(n+2))
         call dcopy(n,ws,1,f(i,1),idf)
  500    continue
      return
      end

      subroutine dftrnx(m,n,f,idf,ws)
c
      integer m,n,idf
      double precision f(idf,*)
      double precision ws(*)
c
c     performs n sine transforms. the transform is unscaled.
c     note that this transform must be scaled up by a factor
c     two in order to correspond to the transform described
c     in chapter 4.2 of numerical solution of the biharmonic
c     equation.
c     m must be odd.(when dsint is called.)
c     workspace ws must have at least int(2.5*m+15) elements.
c     note that routine dsint overwrites f(m+1,j).
c
c     local.
c
c     fourier:            dsinti, dsint   (swarztrauber version 3.)
c
      integer j
      double precision x1
c
      call dsinti(m,ws)
      do 500 j=1,n
         x1       = f(m+1,j)
         call dsint(m,f(1,j),ws)
         f(m+1,j) = x1
  500    continue
      return
      end

      subroutine dsint (n,x,wsave)
      double precision x(*), wsave(*), sqrt3, t1, t2, x1, xh, xim1
      data sqrt3 /  1.7320508075 6887729352 7446341505 87237d0/
c
      if (n-2) 101,102,103
  101 x(1) = x(1)+x(1)
      return
c
  102 xh = sqrt3*(x(1)+x(2))
      x(2) = sqrt3*(x(1)-x(2))
      x(1) = xh
      return
c
  103 np1 = n+1
      ns2 = n/2
      x1 = x(1)
      x(1) = 0.d0
      do 104 k=1,ns2
         kc = np1-k
         t1 = x1-x(kc)
         t2 = wsave(k)*(x1+x(kc))
         x1 = x(k+1)
         x(k+1) = t1+t2
         x(kc+1) = t2-t1
  104 continue
      modn = mod(n,2)
      if (modn .ne. 0) x(ns2+2) = 4.d0*x1
c
      call drfftf (np1,x,wsave(ns2+1))
c
      x(1) = .5d0*x(1)
      do 105 i=3,n,2
         xim1 = x(i-1)
         x(i-1) = -x(i)
         x(i) = x(i-2)+xim1
  105 continue
      if (modn.eq.0) x(n) = -x(n+1)
c
      return
      end

      subroutine drfftf (n,r,wsave)
      double precision r(1), wsave(1)
c
      if (n .eq. 1) return
c
      call drftf1 (n,r,wsave,wsave(n+1),wsave(2*n+1))
c
      return
      end


      subroutine drftf1 (n,c,ch,wa,ifac)
      double precision c(*), ch(*), wa(*)
      integer ifac(*)
c
      nf = ifac(2)
      na = 1
      l2 = n
      iw = n
      do 111 k1=1,nf
         kh = nf-k1
         ip = ifac(kh+3)
         l1 = l2/ip
         ido = n/l2
         idl1 = ido*l1
         iw = iw-(ip-1)*ido
         na = 1-na
         if (ip .ne. 4) go to 102
c
         ix2 = iw+ido
         ix3 = ix2+ido
         if (na .ne. 0) go to 101
         call dradf4 (ido,l1,c,ch,wa(iw),wa(ix2),wa(ix3))
         go to 110
  101    call dradf4 (ido,l1,ch,c,wa(iw),wa(ix2),wa(ix3))
         go to 110
c
  102    if (ip .ne. 2) go to 104
         if (na .ne. 0) go to 103
         call dradf2 (ido,l1,c,ch,wa(iw))
         go to 110
  103    call dradf2 (ido,l1,ch,c,wa(iw))
         go to 110
c
  104    if (ip .ne. 3) go to 106
         ix2 = iw+ido
         if (na .ne. 0) go to 105
         call dradf3 (ido,l1,c,ch,wa(iw),wa(ix2))
         go to 110
  105    call dradf3 (ido,l1,ch,c,wa(iw),wa(ix2))
         go to 110
c
  106    if (ip .ne. 5) go to 108
         ix2 = iw+ido
         ix3 = ix2+ido
         ix4 = ix3+ido
         if (na .ne. 0) go to 107
         call dradf5 (ido,l1,c,ch,wa(iw),wa(ix2),wa(ix3),wa(ix4))
         go to 110
  107    call dradf5 (ido,l1,ch,c,wa(iw),wa(ix2),wa(ix3),wa(ix4))
         go to 110
c
  108    if (ido .eq. 1) na = 1-na
         if (na .ne. 0) go to 109
         call dradfg (ido,ip,l1,idl1,c,c,c,ch,ch,wa(iw))
         na = 1
         go to 110
  109    call dradfg (ido,ip,l1,idl1,ch,ch,ch,c,c,wa(iw))
         na = 0
c
  110    l2 = l1
  111 continue
c
      if (na .eq. 1) return
      do 112 i=1,n
         c(i) = ch(i)
  112 continue
c
      return
      end

      subroutine dradfg (ido,ip,l1,idl1,cc,c1,c2,ch,ch2,wa)
      double precision cc(ido,ip,l1), c1(ido,l1,ip), c2(idl1,ip),
     1  ch(ido,l1,ip), ch2(idl1,ip), wa(1), ai1, ai2, ar1, ar1h, ar2,
     2  ar2h, arg, dc2, dcp, ds2, dsp, tpi
      data tpi   /  6.2831853071 7958647692 5286766559 00577d0/
c
      arg = tpi/float(ip)
      dcp = cos(arg)
      dsp = sin(arg)
      ipph = (ip+1)/2
      ipp2 = ip+2
      idp2 = ido+2
      nbd = (ido-1)/2
      if (ido .eq. 1) go to 119
      do 101 ik=1,idl1
         ch2(ik,1) = c2(ik,1)
  101 continue
      do 103 j=2,ip
         do 102 k=1,l1
            ch(1,k,j) = c1(1,k,j)
  102    continue
  103 continue
c
      if (nbd .gt. l1) go to 107
      is = -ido
      do 106 j=2,ip
         is = is+ido
         idij = is
         do 105 i=3,ido,2
            idij = idij+2
            do 104 k=1,l1
               ch(i-1,k,j) = wa(idij-1)*c1(i-1,k,j)+wa(idij)*c1(i,k,j)
               ch(i,k,j) = wa(idij-1)*c1(i,k,j)-wa(idij)*c1(i-1,k,j)
  104       continue
  105    continue
  106 continue
      go to 111
c
  107 is = -ido
      do 110 j=2,ip
         is = is+ido
         do 109 k=1,l1
            idij = is
            do 108 i=3,ido,2
               idij = idij+2
               ch(i-1,k,j) = wa(idij-1)*c1(i-1,k,j)+wa(idij)*c1(i,k,j)
               ch(i,k,j) = wa(idij-1)*c1(i,k,j)-wa(idij)*c1(i-1,k,j)
  108       continue
  109    continue
  110 continue
c
  111 if (nbd .lt. l1) go to 115
      do 114 j=2,ipph
         jc = ipp2-j
         do 113 k=1,l1
            do 112 i=3,ido,2
               c1(i-1,k,j) = ch(i-1,k,j)+ch(i-1,k,jc)
               c1(i-1,k,jc) = ch(i,k,j)-ch(i,k,jc)
               c1(i,k,j) = ch(i,k,j)+ch(i,k,jc)
               c1(i,k,jc) = ch(i-1,k,jc)-ch(i-1,k,j)
  112       continue
  113    continue
  114 continue
      go to 121
c
  115 do 118 j=2,ipph
         jc = ipp2-j
         do 117 i=3,ido,2
            do 116 k=1,l1
               c1(i-1,k,j) = ch(i-1,k,j)+ch(i-1,k,jc)
               c1(i-1,k,jc) = ch(i,k,j)-ch(i,k,jc)
               c1(i,k,j) = ch(i,k,j)+ch(i,k,jc)
               c1(i,k,jc) = ch(i-1,k,jc)-ch(i-1,k,j)
  116       continue
  117    continue
  118 continue
      go to 121
c
  119 do 120 ik=1,idl1
         c2(ik,1) = ch2(ik,1)
  120 continue
c
  121 do 123 j=2,ipph
         jc = ipp2-j
         do 122 k=1,l1
            c1(1,k,j) = ch(1,k,j)+ch(1,k,jc)
            c1(1,k,jc) = ch(1,k,jc)-ch(1,k,j)
  122    continue
  123 continue
c
      ar1 = 1.d0
      ai1 = 0.d0
      do 127 l=2,ipph
         lc = ipp2-l
         ar1h = dcp*ar1-dsp*ai1
         ai1 = dcp*ai1+dsp*ar1
         ar1 = ar1h
         do 124 ik=1,idl1
            ch2(ik,l) = c2(ik,1)+ar1*c2(ik,2)
            ch2(ik,lc) = ai1*c2(ik,ip)
  124    continue
         dc2 = ar1
         ds2 = ai1
         ar2 = ar1
         ai2 = ai1
         do 126 j=3,ipph
            jc = ipp2-j
            ar2h = dc2*ar2-ds2*ai2
            ai2 = dc2*ai2+ds2*ar2
            ar2 = ar2h
            do 125 ik=1,idl1
               ch2(ik,l) = ch2(ik,l)+ar2*c2(ik,j)
               ch2(ik,lc) = ch2(ik,lc)+ai2*c2(ik,jc)
  125       continue
  126    continue
  127 continue
c
      do 129 j=2,ipph
         do 128 ik=1,idl1
            ch2(ik,1) = ch2(ik,1)+c2(ik,j)
  128    continue
  129 continue
c
      if (ido .lt. l1) go to 132
      do 131 k=1,l1
         do 130 i=1,ido
            cc(i,1,k) = ch(i,k,1)
  130    continue
  131 continue
      go to 135
c
  132 do 134 i=1,ido
         do 133 k=1,l1
            cc(i,1,k) = ch(i,k,1)
  133    continue
  134 continue
c
  135 do 137 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 136 k=1,l1
            cc(ido,j2-2,k) = ch(1,k,j)
            cc(1,j2-1,k) = ch(1,k,jc)
  136    continue
  137 continue
c
      if (ido .eq. 1) return
      if (nbd .lt. l1) go to 141
      do 140 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 139 k=1,l1
            do 138 i=3,ido,2
               ic = idp2-i
               cc(i-1,j2-1,k) = ch(i-1,k,j)+ch(i-1,k,jc)
               cc(ic-1,j2-2,k) = ch(i-1,k,j)-ch(i-1,k,jc)
               cc(i,j2-1,k) = ch(i,k,j)+ch(i,k,jc)
               cc(ic,j2-2,k) = ch(i,k,jc)-ch(i,k,j)
  138       continue
  139    continue
  140 continue
      return
c
  141 do 144 j=2,ipph
         jc = ipp2-j
         j2 = j+j
         do 143 i=3,ido,2
            ic = idp2-i
            do 142 k=1,l1
               cc(i-1,j2-1,k) = ch(i-1,k,j)+ch(i-1,k,jc)
               cc(ic-1,j2-2,k) = ch(i-1,k,j)-ch(i-1,k,jc)
               cc(i,j2-1,k) = ch(i,k,j)+ch(i,k,jc)
               cc(ic,j2-2,k) = ch(i,k,jc)-ch(i,k,j)
  142       continue
  143    continue
  144 continue
c
      return
      end

      subroutine dradf5 (ido,l1,cc,ch,wa1,wa2,wa3,wa4)
      double precision cc(ido,l1,5), ch(ido,5,l1), wa1(1), wa2(1),
     1  wa3(1), wa4(1), ci2, ci3, ci4, ci5, cr2, cr3, cr4, cr5, di2,
     2  di3, di4, di5, dr2, dr3, dr4, dr5, ti11, ti12, ti2, ti3, ti4,
     3  ti5, tr11, tr12, tr2, tr3, tr4, tr5
      data tr11  /  0.3090169943 7494742410 2293417182 81906d0/
      data ti11  /  0.9510565162 9515357211 6439333379 38214d0/
      data tr12  / -0.8090169943 7494742410 2293417182 81906d0/
      data ti12  /  0.5877852522 9247312916 8705954639 07277d0/
c
      do 101 k=1,l1
         cr2 = cc(1,k,5)+cc(1,k,2)
         ci5 = cc(1,k,5)-cc(1,k,2)
         cr3 = cc(1,k,4)+cc(1,k,3)
         ci4 = cc(1,k,4)-cc(1,k,3)
         ch(1,1,k) = cc(1,k,1)+cr2+cr3
         ch(ido,2,k) = cc(1,k,1)+tr11*cr2+tr12*cr3
         ch(1,3,k) = ti11*ci5+ti12*ci4
         ch(ido,4,k) = cc(1,k,1)+tr12*cr2+tr11*cr3
         ch(1,5,k) = ti12*ci5-ti11*ci4
  101 continue
c
      if (ido .eq. 1) return
      idp2 = ido+2
      do 103 k=1,l1
         do 102 i=3,ido,2
            ic = idp2-i
            dr2 = wa1(i-2)*cc(i-1,k,2)+wa1(i-1)*cc(i,k,2)
            di2 = wa1(i-2)*cc(i,k,2)-wa1(i-1)*cc(i-1,k,2)
            dr3 = wa2(i-2)*cc(i-1,k,3)+wa2(i-1)*cc(i,k,3)
            di3 = wa2(i-2)*cc(i,k,3)-wa2(i-1)*cc(i-1,k,3)
            dr4 = wa3(i-2)*cc(i-1,k,4)+wa3(i-1)*cc(i,k,4)
            di4 = wa3(i-2)*cc(i,k,4)-wa3(i-1)*cc(i-1,k,4)
            dr5 = wa4(i-2)*cc(i-1,k,5)+wa4(i-1)*cc(i,k,5)
            di5 = wa4(i-2)*cc(i,k,5)-wa4(i-1)*cc(i-1,k,5)
            cr2 = dr2+dr5
            ci5 = dr5-dr2
            cr5 = di2-di5
            ci2 = di2+di5
            cr3 = dr3+dr4
            ci4 = dr4-dr3
            cr4 = di3-di4
            ci3 = di3+di4
            ch(i-1,1,k) = cc(i-1,k,1)+cr2+cr3
            ch(i,1,k) = cc(i,k,1)+ci2+ci3
            tr2 = cc(i-1,k,1)+tr11*cr2+tr12*cr3
            ti2 = cc(i,k,1)+tr11*ci2+tr12*ci3
            tr3 = cc(i-1,k,1)+tr12*cr2+tr11*cr3
            ti3 = cc(i,k,1)+tr12*ci2+tr11*ci3
            tr5 = ti11*cr5+ti12*cr4
            ti5 = ti11*ci5+ti12*ci4
            tr4 = ti12*cr5-ti11*cr4
            ti4 = ti12*ci5-ti11*ci4
            ch(i-1,3,k) = tr2+tr5
            ch(ic-1,2,k) = tr2-tr5
            ch(i,3,k) = ti2+ti5
            ch(ic,2,k) = ti5-ti2
            ch(i-1,5,k) = tr3+tr4
            ch(ic-1,4,k) = tr3-tr4
            ch(i,5,k) = ti3+ti4
            ch(ic,4,k) = ti4-ti3
  102    continue
  103 continue
c
      return
      end

      subroutine dradf3 (ido,l1,cc,ch,wa1,wa2)
      double precision cc(ido,l1,3), ch(ido,3,l1), wa1(1), wa2(1),
     1  ci2, cr2, di2, di3, dr2, dr3, taui, taur, ti2, ti3, tr2, tr3
      data taur / -0.5 d0 /
      data taui  /  0.8660254037 8443864676 3723170752 93618d0/
c
      do 101 k=1,l1
         cr2 = cc(1,k,2)+cc(1,k,3)
         ch(1,1,k) = cc(1,k,1)+cr2
         ch(1,3,k) = taui*(cc(1,k,3)-cc(1,k,2))
         ch(ido,2,k) = cc(1,k,1)+taur*cr2
  101 continue
c
      if (ido .eq. 1) return
      idp2 = ido+2
      do 103 k=1,l1
         do 102 i=3,ido,2
            ic = idp2-i
            dr2 = wa1(i-2)*cc(i-1,k,2)+wa1(i-1)*cc(i,k,2)
            di2 = wa1(i-2)*cc(i,k,2)-wa1(i-1)*cc(i-1,k,2)
            dr3 = wa2(i-2)*cc(i-1,k,3)+wa2(i-1)*cc(i,k,3)
            di3 = wa2(i-2)*cc(i,k,3)-wa2(i-1)*cc(i-1,k,3)
            cr2 = dr2+dr3
            ci2 = di2+di3
            ch(i-1,1,k) = cc(i-1,k,1)+cr2
            ch(i,1,k) = cc(i,k,1)+ci2
            tr2 = cc(i-1,k,1)+taur*cr2
            ti2 = cc(i,k,1)+taur*ci2
            tr3 = taui*(di2-di3)
            ti3 = taui*(dr3-dr2)
            ch(i-1,3,k) = tr2+tr3
            ch(ic-1,2,k) = tr2-tr3
            ch(i,3,k) = ti2+ti3
            ch(ic,2,k) = ti3-ti2
  102    continue
  103 continue
c
      return
      end

      subroutine dradf2 (ido,l1,cc,ch,wa1)
      double precision cc(ido,l1,2), ch(ido,2,l1), wa1(1), ti2, tr2
c
      do 101 k=1,l1
         ch(1,1,k) = cc(1,k,1)+cc(1,k,2)
         ch(ido,2,k) = cc(1,k,1)-cc(1,k,2)
  101 continue
c
      if (ido-2) 107,105,102
  102 idp2 = ido+2
      do 104 k=1,l1
         do 103 i=3,ido,2
            ic = idp2-i
            tr2 = wa1(i-2)*cc(i-1,k,2)+wa1(i-1)*cc(i,k,2)
            ti2 = wa1(i-2)*cc(i,k,2)-wa1(i-1)*cc(i-1,k,2)
            ch(i,1,k) = cc(i,k,1)+ti2
            ch(ic,2,k) = ti2-cc(i,k,1)
            ch(i-1,1,k) = cc(i-1,k,1)+tr2
            ch(ic-1,2,k) = cc(i-1,k,1)-tr2
  103    continue
  104 continue
c
      if (mod(ido,2) .eq. 1) return
  105 do 106 k=1,l1
         ch(1,2,k) = -cc(ido,k,2)
         ch(ido,1,k) = cc(ido,k,1)
  106 continue
c
  107 return
      end

      subroutine dradf4 (ido,l1,cc,ch,wa1,wa2,wa3)
      double precision cc(ido,l1,4), ch(ido,4,l1), wa1(1), wa2(1),
     1  wa3(1), ci2, ci3, ci4, cr2, cr3, cr4, hsqt2, ti1, ti2, ti3,
     2  ti4, tr1, tr2, tr3, tr4
      data hsqt2 /   .7071067811 8654752440 0844362104 85 d0 /
c
      do 101 k=1,l1
         tr1 = cc(1,k,2)+cc(1,k,4)
         tr2 = cc(1,k,1)+cc(1,k,3)
         ch(1,1,k) = tr1+tr2
         ch(ido,4,k) = tr2-tr1
         ch(ido,2,k) = cc(1,k,1)-cc(1,k,3)
         ch(1,3,k) = cc(1,k,4)-cc(1,k,2)
  101 continue
c
      if (ido-2) 107,105,102
  102 idp2 = ido+2
      do 104 k=1,l1
         do 103 i=3,ido,2
            ic = idp2-i
            cr2 = wa1(i-2)*cc(i-1,k,2)+wa1(i-1)*cc(i,k,2)
            ci2 = wa1(i-2)*cc(i,k,2)-wa1(i-1)*cc(i-1,k,2)
            cr3 = wa2(i-2)*cc(i-1,k,3)+wa2(i-1)*cc(i,k,3)
            ci3 = wa2(i-2)*cc(i,k,3)-wa2(i-1)*cc(i-1,k,3)
            cr4 = wa3(i-2)*cc(i-1,k,4)+wa3(i-1)*cc(i,k,4)
            ci4 = wa3(i-2)*cc(i,k,4)-wa3(i-1)*cc(i-1,k,4)
            tr1 = cr2+cr4
            tr4 = cr4-cr2
            ti1 = ci2+ci4
            ti4 = ci2-ci4
            ti2 = cc(i,k,1)+ci3
            ti3 = cc(i,k,1)-ci3
            tr2 = cc(i-1,k,1)+cr3
            tr3 = cc(i-1,k,1)-cr3
            ch(i-1,1,k) = tr1+tr2
            ch(ic-1,4,k) = tr2-tr1
            ch(i,1,k) = ti1+ti2
            ch(ic,4,k) = ti1-ti2
            ch(i-1,3,k) = ti4+tr3
            ch(ic-1,2,k) = tr3-ti4
            ch(i,3,k) = tr4+ti3
            ch(ic,2,k) = tr4-ti3
  103    continue
  104 continue
      if (mod(ido,2) .eq. 1) return
  105 continue
c
      do 106 k=1,l1
         ti1 = -hsqt2*(cc(ido,k,2)+cc(ido,k,4))
         tr1 = hsqt2*(cc(ido,k,2)-cc(ido,k,4))
         ch(ido,1,k) = tr1+cc(ido,k,1)
         ch(ido,3,k) = cc(ido,k,1)-tr1
         ch(1,2,k) = ti1-cc(ido,k,3)
         ch(1,4,k) = ti1+cc(ido,k,3)
  106 continue
c
  107 return
      end

      subroutine dsinti (n,wsave)
      double precision wsave(1), dt, fk, pi
      data pi /  3.141592653 5897932384 6264338327 950 d0 /
c
      if (n .le. 1) return
      np1 = n+1
      ns2 = n/2
      dt = pi/dfloat(np1)
      fk = 0.d0
      do 101 k=1,ns2
         fk = fk+1.d0
         wsave(k) = 2.d0*sin(fk*dt)
  101 continue
c
      call drffti (np1,wsave(ns2+1))
c
      return
      end

      subroutine drffti (n,wsave)
      double precision wsave(1)
c
      if (n .eq. 1) return
c
      call drfti1 (n,wsave(n+1),wsave(2*n+1))
c
      return
      end

      subroutine drfti1 (n,wa,ifac)
      double precision wa(*), arg, argh, argld, fi, tpi
      integer ifac(*), ntryh(4)
      data ntryh(1), ntryh(2), ntryh(3), ntryh(4) /4, 2, 3, 5/
      data tpi   /  6.2831853071 7958647692 5286766559 00577d0/
c
      nl = n
      nf = 0
      j = 0
c
  101 j = j+1
      if (j.le.4) ntry = ntryh(j)
      if (j.gt.4) ntry = ntry + 2
  104 nq = nl/ntry
      nr = nl-ntry*nq
      if (nr.ne.0) go to 101
c
  105 nf = nf+1
      ifac(nf+2) = ntry
      nl = nq
      if (ntry .ne. 2) go to 107
      if (nf .eq. 1) go to 107
      do 106 i=2,nf
         ib = nf-i+2
         ifac(ib+2) = ifac(ib+1)
  106 continue
      ifac(3) = 2
  107 if (nl .ne. 1) go to 104
      ifac(1) = n
      ifac(2) = nf
c
      argh = tpi/dfloat(n)
      is = 0
      nfm1 = nf-1
      l1 = 1
      if (nfm1 .eq. 0) return
      do 110 k1=1,nfm1
         ip = ifac(k1+2)
         ld = 0
         l2 = l1*ip
         ido = n/l2
         ipm = ip-1
         do 109 j=1,ipm
            ld = ld+l1
            i = is
            argld = dfloat(ld)*argh
            fi = 0.d0
            do 108 ii=3,ido,2
               i = i+2
               fi = fi+1.d0
               arg = fi*argld
               wa(i-1) = cos(arg)
               wa(i) = sin(arg)
  108       continue
            is = is+ido
  109    continue
c
         l1 = l2
  110 continue
c
      return
      end

      subroutine dstart(m,n,alpha,beta,f,idf,bda,bdb,bdc,bdd,dx,dy,del)
c
      integer m,n,idf
      double precision dx,dy,del,alpha,beta
      double precision bda(*),bdb(*),bdc(*),bdd(*)
      double precision f(idf,*)
c
c     this routine computes the right hand side
c     of the discrete system.
c
c     input
c           m,n,alpha,beta,idf,f,bda,bdb,bdc,bdd,dx,dy,del
c     output
c           f
c
c     local.
c
c     blas:          dscal
c
      integer mp,np
      integer i,j
      double precision twody,twodel,del2,dy4,d1,d2,d3
c
      mp    =m+1
      np    =n+1
      twody =2.0d0*dy
      twodel=2.0d0*del
      del2  =del*del
      dy4   =dy**4.0d0
      twody =2.0d0*dy
      twodel=2.0d0*del
      dy4   =dy*dy*dy*dy
      d1    =twodel+twodel+4.0d0-alpha
      d2    =del*d1
      d3    =2.0d0*dx*del2
c
c     scale right hand side.
c
      do 300 j=2,np
         call dscal(m,dy4,f(2,j),1)
  300    continue
c
c     add in contribution from the boundary.
c
      do 400 i=2,mp
         f(i,2)  =f(i,2)    +d1*f(i,1)  -twodel*
     +           (f(i+1,1)  +f(i-1,1))  -twody*bdc(i-1)
         f(i,3)  =f(i,3)    -f(i,1)
         f(i,n+1)=f(i,n+1)  +d1*f(i,n+2)-twodel*
     +           (f(i+1,n+2)+f(i-1,n+2))-twody*bdd(i-1)
         f(i,n)  =f(i,n)    -f(i,n+2)
  400    continue
      do 500 j=2,np
         f(2,j)  =f(2,j)    +d2*f(1,j)  -twodel*
     +           (f(1,j+1)  +f(1,j-1))  -d3*bda(j-1)
         f(3,j)  =f(3,j)    -del2*f(1,j)
         f(m+1,j)=f(m+1,j)  +d2*f(m+2,j)-twodel*
     +           (f(m+2,j+1)+f(m+2,j-1))-d3*bdb(j-1)
         f(m,j)  =f(m,j)    -del2*f(m+2,j)
  500        continue
      f(2,2)     =f(2,2)    +twodel*f(1,1)
      f(m+1,2)   =f(m+1,2)  +twodel*f(m+2,1)
      f(2,n+1)   =f(2,n+1)  +twodel*f(1,n+2)
      f(m+1,n+1) =f(m+1,n+1)+twodel*f(m+2,n+2)
      return
      end

      subroutine dspfa(ap,n,kpvt,info)
      integer n,kpvt(*),info
      double precision ap(*)
c
c     dspfa factors a real symmetric matrix stored in
c     packed form by elimination with symmetric pivoting.
c
c     to solve  a*x = b , follow dspfa by dspsl.
c     to compute  inverse(a)*c , follow dspfa by dspsl.
c     to compute  determinant(a) , follow dspfa by dspdi.
c     to compute  inertia(a) , follow dspfa by dspdi.
c     to compute  inverse(a) , follow dspfa by dspdi.
c
c     on entry
c
c        ap      real (n*(n+1)/2)
c                the packed form of a symmetric matrix  a .  the
c                columns of the upper triangle are stored sequentially
c                in a one-dimensional array of length  n*(n+1)/2 .
c                see comments below for details.
c
c        n       integer
c                the order of the matrix  a .
c
c     output
c
c        ap      a block diagonal matrix and the multipliers which
c                were used to obtain it stored in packed form.
c                the factorization can be written  a = u*d*trans(u)
c                where  u  is a product of permutation and unit
c                upper triangular matrices , trans(u) is the
c                transpose of  u , and  d  is block diagonal
c                with 1 by 1 and 2 by 2 blocks.
c
c        kpvt    integer(n)
c                an integer vector of pivot indices.
c
c        info    integer
c                = 0  normal value.
c                = k  if the k-th pivot block is singular. this is
c                     not an error condition for this subroutine,
c                     but it does indicate that dspsl or dspdi may
c                     divide by zero if called.
c
c     packed storage
c
c          the following program segment will pack the upper
c          triangle of a symmetric matrix.
c
c                k = 0
c                do 20 j = 1, n
c                   do 10 i = 1, j
c                      k = k + 1
c                      ap(k)  = a(i,j)
c             10    continue
c             20 continue
c
c     linpack. this version dated 08/14/78 .
c     james bunch, univ. calif. san diego, argonne nat. lab.
c
c     subroutines and functions
c
c     blas daxpy,dswap,idamax
c     fortran dabs,dmax1,dsqrt
c
c     internal variables
c
      double precision ak,akm1,bk,bkm1,denom,mulk,mulkm1,t
      double precision absakk,alpha,colmax,rowmax
      integer idamax,ij,ijj,ik,ikm1,im,imax,imaxp1,imim,imj,imk
      integer j,jj,jk,jkm1,jmax,jmim,k,kk,km1,km1k,km1km1,km2,kstep
      logical swap
c
c
c     initialize
c
c     alpha is*used in choosing pivot block size.
      alpha = (1.0d0 + sqrt(17.0d0))/8.0d0
c
      info = 0
c
c     main loop on k, which goes from n to 1.
c
      k = n
      ik = (n*(n - 1))/2
   10 continue
c
c        leave the loop if k=0 or k=1.
c
c     ...exit
         if (k .eq. 0) go to 200
         if (k .gt. 1) go to 20
            kpvt(1) = 1
            if (ap(1) .eq. 0.0d0) info = 1
c     ......exit
            go to 200
   20    continue
c
c        this section of code determines the kind of
c        elimination to be performed.  when it is completed,
c        kstep will be set to the size of the pivot block, and
c        swap will be set to .true. if an interchange is
c        required.
c
         km1 = k - 1
         kk = ik + k
         absakk = abs(ap(kk))
c
c        determine the largest off-diagonal element in
c        column k.
c
         imax = idamax(k-1,ap(ik+1),1)
         imk = ik + imax
         colmax = abs(ap(imk))
         if (absakk .lt. alpha*colmax) go to 30
            kstep = 1
            swap = .false.
         go to 90
   30    continue
c
c           determine the largest off-diagonal element in
c           row imax.
c
            rowmax = 0.0d0
            imaxp1 = imax + 1
            im = imax*(imax - 1)/2
            imj = im + 2*imax
            do 40 j = imaxp1, k
               rowmax = max1(rowmax,abs(ap(imj)))
               imj = imj + j
   40       continue
            if (imax .eq. 1) go to 50
               jmax = idamax(imax-1,ap(im+1),1)
               jmim = jmax + im
               rowmax = max1(rowmax,abs(ap(jmim)))
   50       continue
            imim = imax + im
            if (abs(ap(imim)) .lt. alpha*rowmax) go to 60
               kstep = 1
               swap = .true.
            go to 80
   60       continue
            if (absakk .lt. alpha*colmax*(colmax/rowmax)) go to 70
               kstep = 1
               swap = .false.
            go to 80
   70       continue
               kstep = 2
               swap = imax .ne. km1
   80       continue
   90    continue
         if (max1(absakk,colmax) .ne. 0.0d0) go to 100
c
c           column k is zero.  set info and iterate the loop.
c
            kpvt(k) = k
            info = k
         go to 190
  100    continue
         if (kstep .eq. 2) go to 140
c
c           1 x 1 pivot block.
c
            if (.not.swap) go to 120
c
c              perform an interchange.
c
               call dswap(imax,ap(im+1),1,ap(ik+1),1)
               imj = ik + imax
               do 110 jj = imax, k
                  j = k + imax - jj
                  jk = ik + j
                  t = ap(jk)
                  ap(jk) = ap(imj)
                  ap(imj) = t
                  imj = imj - (j - 1)
  110          continue
  120       continue
c
c           perform the elimination.
c
            ij = ik - (k - 1)
            do 130 jj = 1, km1
               j = k - jj
               jk = ik + j
               mulk = -ap(jk)/ap(kk)
               t = mulk
               call daxpy(j,t,ap(ik+1),1,ap(ij+1),1)
               ijj = ij + j
               ap(jk) = mulk
               ij = ij - (j - 1)
  130       continue
c
c           set the pivot array.
c
            kpvt(k) = k
            if (swap) kpvt(k) = imax
         go to 190
  140    continue
c
c           2 x 2 pivot block.
c
            km1k = ik + k - 1
            ikm1 = ik - (k - 1)
            if (.not.swap) go to 160
c
c              perform an interchange.
c
               call dswap(imax,ap(im+1),1,ap(ikm1+1),1)
               imj = ikm1 + imax
               do 150 jj = imax, km1
                  j = km1 + imax - jj
                  jkm1 = ikm1 + j
                  t = ap(jkm1)
                  ap(jkm1) = ap(imj)
                  ap(imj) = t
                  imj = imj - (j - 1)
  150          continue
               t = ap(km1k)
               ap(km1k) = ap(imk)
               ap(imk) = t
  160       continue
c
c           perform the elimination.
c
            km2 = k - 2
            if (km2 .eq. 0) go to 180
               ak = ap(kk)/ap(km1k)
               km1km1 = ikm1 + k - 1
               akm1 = ap(km1km1)/ap(km1k)
               denom = 1.0d0 - ak*akm1
               ij = ik - (k - 1) - (k - 2)
               do 170 jj = 1, km2
                  j = km1 - jj
                  jk = ik + j
                  bk = ap(jk)/ap(km1k)
                  jkm1 = ikm1 + j
                  bkm1 = ap(jkm1)/ap(km1k)
                  mulk = (akm1*bk - bkm1)/denom
                  mulkm1 = (ak*bkm1 - bk)/denom
                  t = mulk
                  call daxpy(j,t,ap(ik+1),1,ap(ij+1),1)
                  t = mulkm1
                  call daxpy(j,t,ap(ikm1+1),1,ap(ij+1),1)
                  ap(jk) = mulk
                  ap(jkm1) = mulkm1
                  ijj = ij + j
                  ij = ij - (j - 1)
  170          continue
  180       continue
c
c           set the pivot array.
c
            kpvt(k) = 1 - k
            if (swap) kpvt(k) = -imax
            kpvt(k-1) = kpvt(k)
  190    continue
         ik = ik - (k - 1)
         if (kstep .eq. 2) ik = ik - (k - 2)
         k = k - kstep
      go to 10
  200 continue
      return
      end
      subroutine  dswap (n,dx,incx,dy,incy)
c
c     interchanges two vectors.
c     uses unrolled loops for increments equal one.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(1),dy(1),dtemp
      integer i,incx,incy,ix,iy,m,mp1,n
c
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c       code for unequal increments or equal increments not equal
c         to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dtemp = dx(ix)
        dx(ix) = dy(iy)
        dy(iy) = dtemp
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c       code for both increments equal to 1
c
c
c       clean-up loop
c
   20 m = mod(n,3)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dtemp = dx(i)
        dx(i) = dy(i)
        dy(i) = dtemp
   30 continue
      if( n .lt. 3 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,3
        dtemp = dx(i)
        dx(i) = dy(i)
        dy(i) = dtemp
        dtemp = dx(i + 1)
        dx(i + 1) = dy(i + 1)
        dy(i + 1) = dtemp
        dtemp = dx(i + 2)
        dx(i + 2) = dy(i + 2)
        dy(i + 2) = dtemp
   50 continue
      return
      end

      integer function idamax(n,dx,incx)
c
c     finds the index of element having max. absolute value.
c     jack dongarra, linpack, 3/11/78.
c     modified to correct problem with negative increment, 8/21/90.
c
      double precision dx(1),dmax
      integer i,incx,ix,n
c
      idamax = 0
      if( n .lt. 1 ) return
      idamax = 1
      if(n.eq.1)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      ix = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      dmax = abs(dx(ix))
      ix = ix + incx
      do 10 i = 2,n
         if(abs(dx(ix)).le.dmax) go to 5
         idamax = i
         dmax = abs(dx(ix))
    5    ix = ix + incx
   10 continue
      return
c
c        code for increment equal to 1
c
   20 dmax = abs(dx(1))
      do 30 i = 2,n
         if(abs(dx(i)).le.dmax) go to 30
         idamax = i
         dmax = abs(dx(i))
   30 continue
      return
      end

      subroutine dspsl(ap,n,kpvt,b)
      integer n,kpvt(1)
      double precision ap(1),b(1)
c
c     dsisl solves the real symmetric system
c     a * x = b
c     using the factors computed by dspfa.
c
c     on entry
c
c        ap      real(n*(n+1)/2)
c                the output from dspfa.
c
c        n       integer
c                the order of the matrix  a .
c
c        kpvt    integer(n)
c                the pivot vector from dspfa.
c
c        b       real(n)
c                the right hand side vector.
c
c     on return
c
c        b       the solution vector  x .
c
c     error condition
c
c        a division by zero may occur if  dspco  has set rcond .eq. 0.0
c        or  dspfa  has set info .ne. 0  .
c
c     to compute  inverse(a) * c  where  c  is a matrix
c     with  p  columns
c           call dspfa(ap,n,kpvt,info)
c           if (info .ne. 0) go to ...
c           do 10 j = 1, p
c              call dspsl(ap,n,kpvt,c(1,j))
c        10 continue
c
c     linpack. this version dated 08/14/78 .
c     james bunch, univ. calif. san diego, argonne nat. lab.
c
c     subroutines and functions
c
c     blas daxpy,ddot
c     fortran iabs
c
c     internal variables.
c
      double precision ak,akm1,bk,bkm1,ddot,denom,temp
      integer ik,ikm1,ikp1,k,kk,km1k,km1km1,kp
c
c     loop backward applying the transformations and
c     d inverse to b.
c
      k = n
      ik = (n*(n - 1))/2
   10 if (k .eq. 0) go to 80
         kk = ik + k
         if (kpvt(k) .lt. 0) go to 40
c
c           1 x 1 pivot block.
c
            if (k .eq. 1) go to 30
               kp = kpvt(k)
               if (kp .eq. k) go to 20
c
c                 interchange.
c
                  temp = b(k)
                  b(k) = b(kp)
                  b(kp) = temp
   20          continue
c
c              apply the transformation.
c
               call daxpy(k-1,b(k),ap(ik+1),1,b(1),1)
   30       continue
c
c           apply d inverse.
c
            b(k) = b(k)/ap(kk)
            k = k - 1
            ik = ik - k
         go to 70
   40    continue
c
c           2 x 2 pivot block.
c
            ikm1 = ik - (k - 1)
            if (k .eq. 2) go to 60
               kp = iabs(kpvt(k))
               if (kp .eq. k - 1) go to 50
c
c                 interchange.
c
                  temp = b(k-1)
                  b(k-1) = b(kp)
                  b(kp) = temp
   50          continue
c
c              apply the transformation.
c
               call daxpy(k-2,b(k),ap(ik+1),1,b(1),1)
               call daxpy(k-2,b(k-1),ap(ikm1+1),1,b(1),1)
   60       continue
c
c           apply d inverse.
c
            km1k = ik + k - 1
            kk = ik + k
            ak = ap(kk)/ap(km1k)
            km1km1 = ikm1 + k - 1
            akm1 = ap(km1km1)/ap(km1k)
            bk = b(k)/ap(km1k)
            bkm1 = b(k-1)/ap(km1k)
            denom = ak*akm1 - 1.0d0
            b(k) = (akm1*bk - bkm1)/denom
            b(k-1) = (ak*bkm1 - bk)/denom
            k = k - 2
            ik = ik - (k + 1) - k
   70    continue
      go to 10
   80 continue
c
c     loop forward applying the transformations.
c
      k = 1
      ik = 0
   90 if (k .gt. n) go to 160
         if (kpvt(k) .lt. 0) go to 120
c
c           1 x 1 pivot block.
c
            if (k .eq. 1) go to 110
c
c              apply the transformation.
c
               b(k) = b(k) + ddot(k-1,ap(ik+1),1,b(1),1)
               kp = kpvt(k)
               if (kp .eq. k) go to 100
c
c                 interchange.
c
                  temp = b(k)
                  b(k) = b(kp)
                  b(kp) = temp
  100          continue
  110       continue
            ik = ik + k
            k = k + 1
         go to 150
  120    continue
c
c           2 x 2 pivot block.
c
            if (k .eq. 1) go to 140
c
c              apply the transformation.
c
               b(k) = b(k) + ddot(k-1,ap(ik+1),1,b(1),1)
               ikp1 = ik + k
               b(k+1) = b(k+1) + ddot(k-1,ap(ikp1+1),1,b(1),1)
               kp = iabs(kpvt(k))
               if (kp .eq. k) go to 130
c
c                 interchange.
c
                  temp = b(k)
                  b(k) = b(kp)
                  b(kp) = temp
  130          continue
  140       continue
            ik = ik + k + k + 1
            k = k + 2
  150    continue
      go to 90
  160 continue
      return
      end
      subroutine dppfa(ap,n,info)
      integer n,info
      double precision ap(1)
c
c     dppfa factors a real symmetric positive definite
c     matrix stored in packed form.
c
c     dppfa is usually called by dppco, but it can be called
c     directly with a saving in time if  rcond  is not needed.
c     (time for dppco) = (1 + 18/n)*(time for dppfa) .
c
c     on entry
c
c        ap      real (n*(n+1)/2)
c                the packed form of a symmetric matrix  a .  the
c                columns of the upper triangle are stored sequentially
c                in a one-dimensional array of length  n*(n+1)/2 .
c                see comments below for details.
c
c        n       integer
c                the order of the matrix  a .
c
c     on return
c
c        ap      an upper triangular matrix  r , stored in packed
c                form, so that  a = trans(r)*r .
c
c        info    integer
c                = 0  for normal return.
c                = k  if the leading minor of order  k  is not
c                     positive definite.
c
c
c     packed storage
c
c          the following program segment will pack the upper
c          triangle of a symmetric matrix.
c
c                k = 0
c                do 20 j = 1, n
c                   do 10 i = 1, j
c                      k = k + 1
c                      ap(k) = a(i,j)
c             10    continue
c             20 continue
c
c     linpack.  this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas ddot
c     fortran dsqrt
c
c     internal variables
c
      double precision ddot,t
      double precision s
      integer j,jj,jm1,k,kj,kk
c     begin block with ...exits to 40
c
c
         jj = 0
         do 30 j = 1, n
            info = j
            s = 0.0d0
            jm1 = j - 1
            kj = jj
            kk = 0
            if (jm1 .lt. 1) go to 20
            do 10 k = 1, jm1
               kj = kj + 1
               t = ap(kj) - ddot(k-1,ap(kk+1),1,ap(jj+1),1)
               kk = kk + k
               t = t/ap(kk)
               ap(kj) = t
               s = s + t*t
   10       continue
   20       continue
            jj = jj + j
            s = ap(jj) - s
c     ......exit
            if (s .le. 0.0d0) go to 40
            ap(jj) = sqrt(s)
   30    continue
         info = 0
   40 continue
      return
      end
      subroutine dppsl(ap,n,b)
      integer n
      double precision ap(1),b(1)
c
c     dppsl solves the real symmetric positive definite
c     system a * x = b
c     using the factors computed by dppco or dppfa.
c
c     on entry
c
c        ap      real (n*(n+1)/2)
c                the output from dppco or dppfa.
c
c        n       integer
c                the order of the matrix  a .
c
c        b       real(n)
c                the right hand side vector.
c
c     on return
c
c        b       the solution vector  x .
c
c     error condition
c
c        a division by zero will occur if the input factor contains
c        a zero on the diagonal.  technically this indicates
c        singularity but it is usually caused by improper subroutine
c        arguments.  it will not occur if the subroutines are called
c        correctly and  info .eq. 0 .
c
c     to compute  inverse(a) * c  where  c  is a matrix
c     with  p  columns
c           call dppco(ap,n,rcond,z,info)
c           if (rcond is too small .or. info .ne. 0) go to ...
c           do 10 j = 1, p
c              call dppsl(ap,n,c(1,j))
c        10 continue
c
c     linpack.  this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     blas daxpy,ddot
c
c     internal variables
c
      double precision ddot,t
      integer k,kb,kk
c
      kk = 0
      do 10 k = 1, n
         t = ddot(k-1,ap(kk+1),1,b(1),1)
         kk = kk + k
         b(k) = (b(k) - t)/ap(kk)
   10 continue
      do 20 kb = 1, n
         k = n + 1 - kb
         b(k) = b(k)/ap(kk)
         kk = kk - k
         t = -b(k)
         call daxpy(k-1,t,ap(kk+1),1,b(1),1)
   20 continue
      return
      end
      double precision function ddot(n,dx,incx,dy,incy)
c
c     forms the dot product of two vectors.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(1),dy(1),dtemp
      integer i,incx,incy,ix,iy,m,mp1,n
c
      ddot = 0.0d0
      dtemp = 0.0d0
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dtemp = dtemp + dx(ix)*dy(iy)
        ix = ix + incx
        iy = iy + incy
   10 continue
      ddot = dtemp
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,5)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dtemp = dtemp + dx(i)*dy(i)
   30 continue
      if( n .lt. 5 ) go to 60
   40 mp1 = m + 1
      do 50 i = mp1,n,5
        dtemp = dtemp + dx(i)*dy(i) + dx(i + 1)*dy(i + 1) +
     *   dx(i + 2)*dy(i + 2) + dx(i + 3)*dy(i + 3) + dx(i + 4)*dy(i + 4)
   50 continue
   60 ddot = dtemp
      return
      end

      subroutine daxpy(n,da,dx,incx,dy,incy)
c
c     constant times a vector plus a vector.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(1),dy(1),da
      integer i,incx,incy,ix,iy,m,mp1,n
c
      if(n.le.0)return
      if (da .eq. 0.0d0) return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dy(iy) + da*dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,4)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dy(i) = dy(i) + da*dx(i)
   30 continue
      if( n .lt. 4 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,4
        dy(i) = dy(i) + da*dx(i)
        dy(i + 1) = dy(i + 1) + da*dx(i + 1)
        dy(i + 2) = dy(i + 2) + da*dx(i + 2)
        dy(i + 3) = dy(i + 3) + da*dx(i + 3)
   50 continue
      return
      end

      subroutine  dcopy(n,dx,incx,dy,incy)
c
c     copies a vector, x, to a vector, y.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(1),dy(1)
      integer i,incx,incy,ix,iy,m,mp1,n
c
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,7)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dy(i) = dx(i)
   30 continue
      if( n .lt. 7 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,7
        dy(i) = dx(i)
        dy(i + 1) = dx(i + 1)
        dy(i + 2) = dx(i + 2)
        dy(i + 3) = dx(i + 3)
        dy(i + 4) = dx(i + 4)
        dy(i + 5) = dx(i + 5)
        dy(i + 6) = dx(i + 6)
   50 continue
      return
      end
      subroutine  dscal(n,da,dx,incx)
c
c     scales a vector by a constant.
c     uses unrolled loops for increment equal to one.
c     jack dongarra, linpack, 3/11/78.
c     modified to correct problem with negative increment, 8/21/90.
c
      double precision da,dx(1)
      integer i,incx,ix,m,mp1,n
c
      if(n.le.0)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      ix = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      do 10 i = 1,n
        dx(ix) = da*dx(ix)
        ix = ix + incx
   10 continue
      return
c
c        code for increment equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,5)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dx(i) = da*dx(i)
   30 continue
      if( n .lt. 5 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,5
        dx(i) = da*dx(i)
        dx(i + 1) = da*dx(i + 1)
        dx(i + 2) = da*dx(i + 2)
        dx(i + 3) = da*dx(i + 3)
        dx(i + 4) = da*dx(i + 4)
   50 continue
      return
      end

c     --------------------------------------------------
c     --------------------------------------------------
c     --------------------------------------------------
c     --------------------------------------------------
c     --------------------------------------------------
c     --------------------------------------------------
c     --------------------------------------------------
