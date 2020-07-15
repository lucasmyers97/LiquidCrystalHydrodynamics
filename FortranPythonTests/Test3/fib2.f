C FILE: FIB2.F
      SUBROUTINE FIBADD(A,N)
C
C     ADD 1 TO THE FIBONACCI ROUTINE
C
      INTEGER N
      REAL*8 A(N)
      CALL FIB(A, N)
      DO I=1,N
          A(I) = A(I) + 1.0D0
      ENDDO
      END
C END FILE FIB2.F