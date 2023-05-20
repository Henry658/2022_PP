#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  float F_nine = 9.999999f;
  __pp_vec_float x;
  __pp_vec_int y;
  __pp_vec_float result = _pp_vset_float(0.f);
  __pp_vec_int V_I_zero = _pp_vset_int(0), V_I_one = _pp_vset_int(1), count;
  __pp_vec_float V_F_zero = _pp_vset_float(0.f), V_F_nine = _pp_vset_float(9.999999f);
  __pp_mask maskAll, mask_is_zero, mask_is_not_zero , exp_mask, gt_ceil;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    if (i + VECTOR_WIDTH > N) maskAll = _pp_init_ones(N-i); // select remain
    else maskAll = _pp_init_ones(); // select all

    mask_is_zero = _pp_init_ones(0);
    exp_mask = _pp_init_ones(0);
    gt_ceil = _pp_init_ones(0);
    
    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // load values

    // Load vector of values from contiguous memory addresses
    _pp_vload_int(y, exponents + i, maskAll); // load exponents

    _pp_veq_int(mask_is_zero, y, V_I_zero, maskAll); // set mask_is_zero

    // if y == 0
    _pp_vset_float(result, 1.f, mask_is_zero); // set to 1.f if mask on

    // else y != 0
    // Inverse maskIsNegative to generate "else" mask
    mask_is_not_zero = _pp_mask_not(mask_is_zero);

    _pp_vmove_float(result, x, mask_is_not_zero); // result = x

    _pp_vsub_int(count, y, V_I_one, mask_is_not_zero); // count = y -1
    _pp_vgt_int(exp_mask, count, V_I_zero, mask_is_not_zero);

    while (_pp_cntbits(exp_mask)) { // count > 0 
      _pp_vmult_float(result, result, x, exp_mask); // result *= x

      _pp_vsub_int(count, count, V_I_one, exp_mask); // count--
      _pp_vgt_int(exp_mask, count, V_I_zero, exp_mask); // update exp_mask
    }

    // if result > 9.999999f
    _pp_vgt_float(gt_ceil, result, V_F_nine, maskAll); // set greate than mask
    _pp_vset_float(result, F_nine, gt_ceil); // result = 9.999999f;

    _pp_vstore_float(output + i, result, maskAll);
  }
  
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  int len;
  float ans = 0.f;
  float *output = new float [VECTOR_WIDTH];
  __pp_vec_float x, result;
  __pp_mask maskAll = _pp_init_ones();
    
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    len = VECTOR_WIDTH;
    
    _pp_vload_float(x, values + i, maskAll);
    while (len > 1) {
      _pp_hadd_float(x, x);
      _pp_interleave_float(x, x);
      len >>= 1;
    }
    _pp_vstore_float(output, x, maskAll);
    ans += output[0];
  }
  
  delete[] output;
  return ans;
}