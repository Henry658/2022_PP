#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<stdio.h>
#include<pthread.h>
#include<immintrin.h>
#include<unistd.h>
using namespace std;


pthread_t* threads;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
long long int number_in_circle, part_number_of_tosses, part_number_in_circle;
__m256 ones, RND_MX;

// ref : https://github.com/lemire/SIMDxorshift
// ref : https://en.wikipedia.org/wiki/Xorshift

/* Keys for scalar xorshift128. Must be non-zero
These are modified by xorshift128plus.
 */
struct avx_xorshift128plus_key_s {
    __m256i part1;
    __m256i part2;
};

typedef struct avx_xorshift128plus_key_s avx_xorshift128plus_key_t;

/* used by xorshift128plus_jump_onkeys */
static void xorshift128plus_onkeys(uint64_t * ps0, uint64_t * ps1) {
	uint64_t s1 = *ps0;
	const uint64_t s0 = *ps1;
	*ps0 = s0;
	s1 ^= s1 << 23; // a
	*ps1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
}

/* used by avx_xorshift128plus_init */
static void xorshift128plus_jump_onkeys(uint64_t in1, uint64_t in2,
		uint64_t * output1, uint64_t * output2) {
	/* todo: streamline */
	static const uint64_t JUMP[] = { 0x8a5cd789635d2dff, 0x121fd2155c472f96 };
	uint64_t s0 = 0;
	uint64_t s1 = 0;
	for (unsigned int i = 0; i < sizeof(JUMP) / sizeof(*JUMP); i++)
		for (int b = 0; b < 64; b++) {
			if (JUMP[i] & 1ULL << b) {
				s0 ^= in1;
				s1 ^= in2;
			}
			xorshift128plus_onkeys(&in1, &in2);
		}
	output1[0] = s0;
	output2[0] = s1;
}

void avx_xorshift128plus_init(uint64_t key1, uint64_t key2,
		avx_xorshift128plus_key_t *key) {
	uint64_t S0[4];
	uint64_t S1[4];
	S0[0] = key1;
	S1[0] = key2;
	xorshift128plus_jump_onkeys(*S0, *S1, S0 + 1, S1 + 1);
	xorshift128plus_jump_onkeys(*(S0 + 1), *(S1 + 1), S0 + 2, S1 + 2);
	xorshift128plus_jump_onkeys(*(S0 + 2), *(S1 + 2), S0 + 3, S1 + 3);
	key->part1 = _mm256_loadu_si256((const __m256i *) S0);
	key->part2 = _mm256_loadu_si256((const __m256i *) S1);
}

/*
 Return a 256-bit random "number"
 */
__m256i avx_xorshift128plus(avx_xorshift128plus_key_t *key) {
	__m256i s1 = key->part1;
	const __m256i s0 = key->part2;
	key->part1 = key->part2;
	s1 = _mm256_xor_si256(key->part2, _mm256_slli_epi64(key->part2, 23));
	key->part2 = _mm256_xor_si256(
			_mm256_xor_si256(_mm256_xor_si256(s1, s0),
					_mm256_srli_epi64(s1, 18)), _mm256_srli_epi64(s0, 5));
	return _mm256_add_epi64(key->part2, s0);
}

static inline void* get_number_in_circle(void *arg) {
	unsigned int seed = time(NULL) + pthread_self();
	int rand_num = rand_r(&seed);
	long long int temp_number_in_circle = 0;
	__m256 _x, _y, mask, add, vcnt;
	__m256i numx, numy;
	float *ans = (float*) _mm_malloc(8*sizeof(float), 32);	
	vcnt = _mm256_setzero_ps();
	avx_xorshift128plus_key_t xkey, ykey;
	// typedef union __declspec(intrin_type) _CRT_ALIGN(32) __m256 {
	//       float m256_f32[8];
	// } __m256

	avx_xorshift128plus_init(4321+(uint64_t)seed, 1234+(uint64_t)seed, &xkey);
	avx_xorshift128plus_init(1234+(uint64_t)seed, 4321+(uint64_t)seed, &ykey);

	for (long long int toss = 0 ; toss < part_number_of_tosses ; toss += 8) {
		numx = avx_xorshift128plus(&xkey); // Return a 256-bit random "number"
		_x = _mm256_cvtepi32_ps(numx); // Converts extended packed 32-bit integer values to packed single-precision floating point values.
		numy = avx_xorshift128plus(&ykey); // Return a 256-bit random "number"
		_y = _mm256_cvtepi32_ps(numy); // Converts extended packed 32-bit integer values to packed single-precision floating point values.

		_x = _mm256_div_ps(_x, RND_MX); // _x[i] = (float)(rand_num & 0xffff) / 0xffff;
		_y = _mm256_div_ps(_y, RND_MX); // _y[i] = (float)(rand_num & 0xffff) / 0xffff;

		_x = _mm256_mul_ps(_x, _x); // _x = _mm256_mul_ps(_x, _x);
		_y = _mm256_mul_ps(_y, _y); // _y = _mm256_mul_ps(_y, _y);

		_x = _mm256_add_ps(_x, _y); // _x = _mm256_mul_ps(_x, _y);

													// https://stackoverflow.com/questions/16988199/how-to-choose-avx-compare-predicate-variants
		mask = _mm256_cmp_ps(_x, ones, _CMP_LE_OQ); // #define _CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
                                                	// #define _CMP_LE_OQ    0x12 /* Less-than-or-equal (ordered, non-signaling)  */
		add = _mm256_and_ps(mask, ones);
		vcnt = _mm256_add_ps(vcnt, add);

		if (toss%1500000 == 0) {
			_mm256_store_ps(ans, vcnt);	
			vcnt = _mm256_setzero_ps();
			for (int i = 0 ; i < 8 ; ++i) {
				if (ans[i]) {
				temp_number_in_circle += ans[i];
				}		
			}
		}
	}
	_mm256_store_ps(ans, vcnt);
	for (int i = 0 ; i < 8 ; ++i) {
		if (ans[i]) {
		temp_number_in_circle += ans[i];
		}		
	}
	pthread_mutex_lock(&mutex);
	number_in_circle += temp_number_in_circle;
	pthread_mutex_unlock(&mutex);
	pthread_exit(EXIT_SUCCESS);
}

double get_pi(int num_threads, long long int N) {
	double pi;
	ones = _mm256_set1_ps(1.0f), RND_MX = _mm256_set1_ps(INT32_MAX);
	number_in_circle = 0;
	part_number_of_tosses = N / num_threads;
    //part_number_of_tosses = part_number_of_tosses - part_number_of_tosses % num_threads;
	for (int i = 0 ; i < num_threads ; ++i ) {
		pthread_create(&threads[i], NULL, get_number_in_circle, NULL);
	}

	for (int i = 0 ; i < num_threads ; ++i ) {
		pthread_join(threads[i], NULL);
		//number_in_circle += (long long int)retrun_value;
	}

	pi = 4.0 * ((double)number_in_circle / N);
	
	return pi;
}

int main(int argc, char *argv[]){
	int num_threads = atoi(argv[1]);
	long long int N = atoll(argv[2]);
	//printf("Number of CPU %ld\n",sysconf(_SC_NPROCESSORS_ONLN));
	threads = new pthread_t [num_threads];

	double pi = get_pi(num_threads, N);

	printf("%.6f\n", pi); // 3.1415926
	delete [] threads;
	return 0;
}
