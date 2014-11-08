__global__ void k_makeRedundant(const cuComplex* input, cuComplex* output, int nRows, int nCols) {
  volatile int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
  volatile int gid_y = threadIdx.y + blockIdx.y * blockDim.y;
  const int nUnique = nCols/2 + 1;
  const int inputPlaneOffset = nRows * nUnique * blockIdx.z;
  const int outputPlaneOffset = nRows * nCols * blockIdx.z;

  // index for reading :
  volatile int gid = gid_x + nUnique * gid_y;

  cuComplex val;

  if(gid_x < nUnique && gid_y < nRows) {
    // write the non redundant part in the new array :
    val = input[gid + inputPlaneOffset];
    gid = gid_x + nCols * gid_y; // new index for writing
    output[gid + outputPlaneOffset] = val;
  }

  // shift'n'flip
  gid_x = nCols - gid_x;

  if(gid_y != 0)
    gid_y = nRows - gid_y;

  gid = gid_x + nCols * gid_y;

  // write conjugate :
  if(gid_x >= nUnique && gid_x < nCols && gid_y >= 0 && gid_y < nRows) {
    val.y = -val.y;
    output[gid+outputPlaneOffset] = val; // never coalesced with compute <= 1.1 ; coalesced if >= 1.2 AND w multiple of 16 AND good call configuration
  }
}



void fill_hermitian_call(const cuComplex* input, cuComplex* output, 
                    const int nPlanes, const int nRows, const int nCols) {
  const int nColsPerBlock = min(nCols, max(128/nCols, 1));
  const int nRowsPerBlock = 1;
  dim3 threads(nRowsPerBlock,nColsPerBlock);
  dim3 blocks(nRows/nRowsPerBlock,nCols/nColsPerBlock,nPlanes);
  k_makeRedundant<<<blocks,threads>>>(input, output, nRows, nCols);
  CUDA_LOOK_FOR_ERROR();
}

