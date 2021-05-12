/* SOURCE: mpi4py */

/* Author:  Lisandro Dalcin */
/* Contact: dalcinl@gmail.com */

/* Copyright (c) 2021, Lisandro Dalcin. */
/* All rights reserved. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in the */
/*   documentation and/or other materials provided with the distribution. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR */
/* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT */
/* HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, */
/* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT */
/* LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, */
/* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY */
/* THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT */
/* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE */
/* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */


static std::string PyMPI_Get_vendor(int         *version_major,
                                    int         *version_minor,
                                    int         *version_micro)
{
  const char *name = "unknown";
  int major=0, minor=0, micro=0;

#if defined(I_MPI_VERSION)

  name = "Intel MPI";
  #if defined(I_MPI_NUMVERSION)
  {int version = I_MPI_NUMVERSION/1000;
  major = version/10000; version -= major*10000;
  minor = version/100;   version -= minor*100;
  micro = version/1;     version -= micro*1; }
  #else
  (void)sscanf(I_MPI_VERSION,"%d.%d Update %d",&major,&minor,&micro);
  #endif

#elif defined(PLATFORM_MPI)

  name = "Platform MPI";
  major = (PLATFORM_MPI>>24)&0xff;
  minor = (PLATFORM_MPI>>16)&0xff;
  micro = (PLATFORM_MPI>> 8)&0xff;
  major = (major/16)*10+(major%16);

#elif defined(MSMPI_VER)

  name = "Microsoft MPI";
  major = MSMPI_VER >> 8;
  minor = MSMPI_VER & 0xFF;

#elif defined(MVAPICH2_VERSION) || defined(MVAPICH2_NUMVERSION)

  name = "MVAPICH2";
  #if defined(MVAPICH2_NUMVERSION)
  {int version = MVAPICH2_NUMVERSION/1000;
  major = version/10000; version -= major*10000;
  minor = version/100;   version -= minor*100;
  micro = version/1;     version -= micro*1; }
  #elif defined(MVAPICH2_VERSION)
  (void)sscanf(MVAPICH2_VERSION,"%d.%d.%d",&major,&minor,&micro);
  #endif

#elif defined(MPICH_NAME) && (MPICH_NAME == 3)

  name = "MPICH";
  #if defined(MPICH_NUMVERSION)
  {int version = MPICH_NUMVERSION/1000;
  major = version/10000; version -= major*10000;
  minor = version/100;   version -= minor*100;
  micro = version/1;     version -= micro*1; }
  #elif defined(MPICH_VERSION)
  (void)sscanf(MPICH_VERSION,"%d.%d.%d",&major,&minor,&micro);
  #endif

#elif defined(MPICH_NAME) && (MPICH_NAME == 2)

  name = "MPICH2";
  #if defined(MPICH2_NUMVERSION)
  {int version = MPICH2_NUMVERSION/1000;
  major = version/10000; version -= major*10000;
  minor = version/100;   version -= minor*100;
  micro = version/1;     version -= micro*1; }
  #elif defined(MPICH2_VERSION)
  (void)sscanf(MPICH2_VERSION,"%d.%d.%d",&major,&minor,&micro);
  #endif

#elif defined(MPICH_NAME) && (MPICH_NAME == 1)

  name = "MPICH1";
  #if defined(MPICH_VERSION)
  (void)sscanf(MPICH_VERSION,"%d.%d.%d",&major,&minor,&micro);
  #endif

#elif defined(OPEN_MPI)

  name = "Open MPI";
  #if defined(OMPI_MAJOR_VERSION)
  major = OMPI_MAJOR_VERSION;
  #endif
  #if defined(OMPI_MINOR_VERSION)
  minor = OMPI_MINOR_VERSION;
  #endif
  #if defined(OMPI_RELEASE_VERSION)
  micro = OMPI_RELEASE_VERSION;
  #endif

  #if defined(OMPI_MAJOR_VERSION)
  #if OMPI_MAJOR_VERSION >= 10
  name = "Spectrum MPI";
  #endif
  #endif

#elif defined(LAM_MPI)

  name = "LAM/MPI";
  #if defined(LAM_MAJOR_VERSION)
  major = LAM_MAJOR_VERSION;
  #endif
  #if defined(LAM_MINOR_VERSION)
  minor = LAM_MINOR_VERSION;
  #endif
  #if defined(LAM_RELEASE_VERSION)
  micro = LAM_RELEASE_VERSION;
  #endif

#endif

  //if (vendor_name)   *vendor_name   = name;
  if (version_major) *version_major = major;
  if (version_minor) *version_minor = minor;
  if (version_micro) *version_micro = micro;

  return name;
}

/*
   Local variables:
   c-basic-offset: 2
   indent-tabs-mode: nil
   End:
*/
