#include <volk/volk.h>
#include <volk/constants.h>
#include <chrono>
#include <random>
#include <iostream>
#include <sstream>

using namespace std::chrono;
using namespace std;

void
gen_indata(float* v, uint32_t num_points, float mu , float omega){
  std::mt19937 eng;  // a core engine class 
  std::normal_distribution<float> dist(mu,omega);

  //chrono::high_resolution_clock clock;
  //clock::duration now = clock::now()
  eng.seed( time(NULL) );

  for (unsigned int i = 0; i < num_points; ++i) {
    v[i] = dist(eng);
  }
}

static inline float update_square_sum_1_val(const float SquareSum,
                                            const float Sum,
                                            const uint32_t len,
                                            const float val)
{
    // Updates a sum of squares calculated over len values with the value val
    float n = (float)len;
    return SquareSum + 1.f / (n * (n + 1.f)) * (n * val - Sum) * (n * val - Sum);
}

static inline float add_square_sums(const float SquareSum0,
                                    const float Sum0,
                                    const float SquareSum1,
                                    const float Sum1,
                                    const uint32_t len)
{
    // Add two sums of squares calculated over the same number of values, len
    float n = (float)len;
    return SquareSum0 + SquareSum1 + .5f / n * (Sum0 - Sum1) * (Sum0 - Sum1);
}

static inline void accrue_result(float* PartialSquareSums,
                                 float* PartialSums,
                                 const uint32_t NumberOfPartitions,
                                 const uint32_t PartitionLen)
{
    // Add all partial sums and square sums into the first element of the arrays
    uint32_t accumulators = NumberOfPartitions;
    uint32_t stages = 0;
    uint32_t offset = 1;
    uint32_t partition_len = PartitionLen;

    while (accumulators >>= 1) {
        stages++;
    } // Integer log2
    accumulators = NumberOfPartitions;

    for (uint32_t s = 0; s < stages; s++) {
        accumulators /= 2;
        uint32_t idx = 0;
        for (uint32_t a = 0; a < accumulators; a++) {
            PartialSquareSums[idx] = add_square_sums(PartialSquareSums[idx],
                                                     PartialSums[idx],
                                                     PartialSquareSums[idx + offset],
                                                     PartialSums[idx + offset],
                                                     partition_len);
            PartialSums[idx] += PartialSums[idx + offset];
            idx += 2 * offset;
        }
        offset *= 2;
        partition_len *= 2;
    }
}

void
twopass_f(float *std, float* mean, const float *v, uint32_t num_points){
  
  *mean = 0.f;
  for (uint32_t i = 0; i < num_points; ++i) {
    *mean += v[i];
  }
  *mean /= num_points;

  *std = 0.f;
  for (uint32_t i = 0; i < num_points; ++i) {
    *std += (v[i]-*mean)*(v[i]-*mean);
  }
  *std /= num_points;
  *std = sqrt(*std);
}

void
twopass_d(double *std, double* mean, const float *v, uint32_t num_points){
  
  double _mean = 0.;
  for (uint32_t i = 0; i < num_points; ++i) {
    _mean += v[i];
  }

  _mean /= num_points;
  *mean = _mean;

  double _std = 0.;
  for (uint32_t i = 0; i < num_points; ++i) {
    _std += (v[i] - _mean)*(v[i] - _mean);
  }

  _std /= num_points;
  *std = std::sqrt(_std);
}


void
volk_std(float *std, float* mean, const float *v, const uint32_t num_points){

  volk_32f_stddev_and_mean_32f_x2(std, mean, v, num_points);
}

void
welford_stdmean_f(float *std, float* mean, const float *v, uint32_t num_points){
  float M = v[0];
  float S = 0.f;

  for (uint32_t i = 1; i < num_points; ++i) {
    float M_old = M;
    float x = v[i];
    M += (x - M)/((float) i + 1);
    S += (x - M)*(x - M_old);
  }

  *std = std::sqrt( S / num_points );
  *mean = M;
}

void
yc_f(float *std, float* mean, const float *v, uint32_t num_points){
  uint32_t N = num_points;

  if (N == 0) { return; }
  *std = 0.f;
  *mean = v[0];
  if (N < 2) { return; }


  uint32_t idx = 1;
  for (uint32_t i = 1; i < N; i++) {
    float val = v[idx++];
    float n = (float) i;
    float n_plus_one = n + 1.f;
    *mean += val;
    *std  += 1.f/(n * n_plus_one) * pow( n_plus_one * val - (*mean), 2);
    //*std  += (r *  ((n + 1) * val - (*mean))) * ((n + 1) * val - (*mean));
  }

  *mean /= N;
  *std  = std::sqrt( *std / N );
}

void
yc2_f(float *std, float* mean, const float *v, uint32_t num_points){
  uint32_t N = num_points;

  if (N == 0) { return; }
  *std  = 0.f;
  *mean = v[0];
  if (N < 2) { return; }

  uint32_t half_points = N / 2;

  //init accumulators
  float Sum[2] = { v[0], v[1] };
  float SquareSum[2] = {0.f, 0.f};

  uint32_t idx = 2;
  for (uint32_t i = 1; i < half_points; i++) {
    float v0 = v[idx++];
    float v1 = v[idx++];

    Sum[0] += v0;
    Sum[1] += v1;

    float n = (float)i;
    float n_plus_one = n + 1.f;
    float r = 1.f/(n * n_plus_one);

    SquareSum[0] += r * pow( n_plus_one * v0 - Sum[0], 2);
    SquareSum[1] += r * pow( n_plus_one * v1 - Sum[1], 2);
  }

  float S = Sum[0] + Sum[1];
  float SS = SquareSum[0] + SquareSum[1] + .5f / half_points * pow( Sum[0] - Sum[1], 2);

  uint32_t points_done = half_points * 2;
  for (; points_done < N; points_done++) {
    float r = 1.f / ( points_done * (points_done + 1) );
    float v0 = v[idx++];
    S  += v0;
    SS += r * pow( (points_done + 1) * v0 - S, 2);
  }  

  *mean = S / N;
  *std  = std::sqrt( SS / N );
}

void
yc4_f(float *std, float* mean, const float *v, uint32_t num_points){
  uint32_t qtr_points = num_points / 4;
  const float* in_ptr = v;

  //init accumulators
  float T0 = (*in_ptr++);
  float T1 = (*in_ptr++);
  float T2 = (*in_ptr++);
  float T3 = (*in_ptr++);
  float S0 = 0.f;
  float S1 = 0.f;
  float S2 = 0.f;
  float S3 = 0.f;

  uint32_t number = 1;
  for (; number < qtr_points; number++)
  {
    float factor = 1.f/( number*(number + 1) );
    float number_plus_one = number + 1.f;

    float v0 = (*in_ptr++);
    T0 += v0;
    S0 += factor*( number_plus_one*v0 - T0 )*( number_plus_one*v0 - T0 );

    float v1 = (*in_ptr++);
    T1 += v1;
    S1 += factor*( number_plus_one*v1 - T1 )*( number_plus_one*v1 - T1 );

    float v2 = (*in_ptr++);
    T2 += v2;
    S2 += factor*( number_plus_one*v2 - T2 )*( number_plus_one*v2 - T2 );

    float v3 = (*in_ptr++);
    T3 += v3;
    S3 += factor*( number_plus_one*v3 - T3 )*( number_plus_one*v3 - T3 );        
  }

  float T_tot = (T0 + T1) + (T2 + T3);
  float S01 = S0 + S1 + 1.f/(2*qtr_points)*( T0 - T1 )*( T0 - T1 );
  float S23 = S2 + S3 + 1.f/(2*qtr_points)*( T2 - T3 )*( T2 - T3 );
  float S_tot = S01 + S23 + 1.f/(4*qtr_points)*( (T0+T1) - (T2+T3) )*( (T0+T1) - (T2+T3) );

  *mean = T_tot / num_points;
  *std  = std::sqrt( S_tot / num_points );
}

void
yc8_f(float *std, float* mean, const float *v, uint32_t num_points){
  uint32_t eigth_points = num_points / 8;
  const float* in_ptr = v;

  //init accumulators
  float T0 = (*in_ptr++);
  float T1 = (*in_ptr++);
  float T2 = (*in_ptr++);
  float T3 = (*in_ptr++);
  float T4 = (*in_ptr++);
  float T5 = (*in_ptr++);
  float T6 = (*in_ptr++);
  float T7 = (*in_ptr++);  
  float S0 = 0.f;
  float S1 = 0.f;
  float S2 = 0.f;
  float S3 = 0.f;
  float S4 = 0.f;
  float S5 = 0.f;
  float S6 = 0.f;
  float S7 = 0.f;

  uint32_t number = 1;
  for (; number < eigth_points; number++)
  {
    float factor = 1.f/( number*(number + 1) );
    float number_plus_one = number + 1.f;

    float v0 = (*in_ptr++);
    T0 += v0;
    S0 += factor*( number_plus_one*v0 - T0 )*( number_plus_one*v0 - T0 );

    float v1 = (*in_ptr++);
    T1 += v1;
    S1 += factor*( number_plus_one*v1 - T1 )*( number_plus_one*v1 - T1 );

    float v2 = (*in_ptr++);
    T2 += v2;
    S2 += factor*( number_plus_one*v2 - T2 )*( number_plus_one*v2 - T2 );

    float v3 = (*in_ptr++);
    T3 += v3;
    S3 += factor*( number_plus_one*v3 - T3 )*( number_plus_one*v3 - T3 );   

    float v4 = (*in_ptr++);
    T4 += v4;
    S4 += factor*( number_plus_one*v4 - T4 )*( number_plus_one*v4 - T4 );   

    float v5 = (*in_ptr++);
    T5 += v5;
    S5 += factor*( number_plus_one*v5 - T5 )*( number_plus_one*v5 - T5 );   

    float v6 = (*in_ptr++);
    T6 += v6;
    S6 += factor*( number_plus_one*v6 - T6 )*( number_plus_one*v6 - T6 );   

    float v7 = (*in_ptr++);
    T7 += v7;
    S7 += factor*( number_plus_one*v7 - T7 )*( number_plus_one*v7 - T7 );                        
  }

  float T_tot = ((T0 + T1) + (T2 + T3)) + ((T4 + T5) + (T6 + T7));
  float S01 = S0 + S1 + 1.f/(2*eigth_points)*( T0 - T1 )*( T0 - T1 );
  float S23 = S2 + S3 + 1.f/(2*eigth_points)*( T2 - T3 )*( T2 - T3 );
  float S45 = S4 + S5 + 1.f/(2*eigth_points)*( T4 - T5 )*( T4 - T5 );
  float S67 = S6 + S7 + 1.f/(2*eigth_points)*( T6 - T7 )*( T6 - T7 );

  float S0123 = S01 + S23 + 1.f/(4*eigth_points)*( (T0+T1) - (T2+T3) )*( (T0+T1) - (T2+T3) );
  float S4567 = S45 + S67 + 1.f/(4*eigth_points)*( (T4+T5) - (T6+T7) )*( (T4+T5) - (T6+T7) );

  float S_tot = S0123 + S4567 + 1.f/num_points*
    ( (T0+T1+T2+T3) - (T4+T5+T6+T7) )*( (T0+T1+T2+T3) - (T4+T5+T6+T7) );


  *mean = T_tot/num_points;
  *std  = sqrt( S_tot/num_points );
}

void
naive_f(float* stddev, float* mean, const float *v, uint32_t num_points)
{
  float _std = 0.f;
  float _mean = 0.f;

  for(uint32_t i = 0; i < num_points; i++){
    _std  += v[i] * v[i];
    _mean += v[i];
  }
  _mean /= num_points;
  _std  /= num_points;
  _std -= (_mean * _mean);
  _std = sqrt(_std);
  *stddev = _std;
  *mean = _mean;
}

void 
generic_f(float* stddev, float* mean, const float *v, uint32_t num_points) {
  float Sum = v[0];
  float SquareSum = 0.f;

  for (uint32_t i = 1; i < num_points; i++) {
    float n = (float) i;
    float n_plus_one = n + 1.f;
    Sum += v[i];
    SquareSum += 1.f / (n * n_plus_one) * pow(n_plus_one * v[i] - Sum, 2);
  }
  *stddev = sqrt(SquareSum / num_points);
  *mean = Sum / num_points;
}

void
test_it() { 

}

int main(int argc, char* argv[]){

  setlocale(LC_NUMERIC, "en_US.UTF-8");

    if (argc == 1) {
    std::cout << "Usage: " << argv[0] << " " << std::endl;
    std::cout << "  VOLK        : " << volk_version() << std::endl;
    std::cout << "    alignment : " << volk_get_alignment() << std::endl;
    std::string ms = volk_available_machines();
    std::stringstream ss(ms);
    std::vector<std::string> machines;
    while (std::getline(ss, ms, ';')) {
        machines.push_back(ms);
    }
    std::cout << "    machines  : " << std::endl;
    for (auto m : machines) {
    std::cout << "           ⌞ " << m << std::endl;
    }
    
    return 0;
  }

  //const char* mac = volk_available_machines();
  //const int alignment = volk_get_alignment();

  int N = 10;
  int iters = 1;
  float mean = 10.f;
  float stddev = 1.f;


  //high_resolution_clock::time_point t1,t2;


  if(argc>1){
    N = atoi(argv[1]);
    if (N==0) {N=1000;}
  }
  int na = volk_get_alignment();
  float* in = (float*) volk_malloc(N*sizeof(float), na);

  if(argc>2){
    iters = atoi(argv[2]);
    if (iters==0) {iters=1;}
  }

  if(argc>3){
    mean = atof(argv[3]);
  }

  if(argc>4){
    stddev = atof(argv[4]);
  }  

  //cout << "VOLK machines: " << mac << endl;
  cout << "vlen  : " << N << endl;
  cout << "iters : " << iters << endl;
  cout << "mean  : " << mean << endl;
  cout << "stddev: " << stddev << endl;

  gen_indata(in, N, mean, stddev);
  if (N<=10){
    //printz( "indata", in);
  }

  cout << ("\nNAME              TIME        MEAN    RELERR (dB)   STDDEV     REL ERR (dB)\n");
  cout << ("---------------------------------------------------------------------------\n");
  const char* fmt =  "%9.2f  %12.8f   %6.1f  %12.8f     %6.1f\n";
  const char* fmtd =  "%9.2f  %12.8f     N/A   %12.8f       N/A\n";

  float mu, omega;
  double oerr = -99., merr = -99., oerrmax = -200., merrmax = -200.;
  double mud, omegad;


  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    twopass_d(&omegad, &mud, in, N);
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("2-pass double ");
  printf(fmtd, duration, mud, omegad);

  /*
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    twopass_f(&omega, &mu, in);
  }
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  merr = abs((double)mu - mud) < 1.e-7 ? -99. : 10.*log10( abs(( (double)mu - mud) / mud) );
  oerr = abs((double)omega - omegad) < 1.e-7 ? -99. : 10.*log10(abs((double)omega - omegad) / omegad);

  printf("2-pass float  ");
  printf(fmt, duration, mu, merr, omega, oerr);

  */
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    naive_f(&omega, &mu, in, N);
    merr = abs((double)mu - mud) < 1.e-10 ? -99. : 10.*log10( abs(( (double)mu - mud) / mud) );
    oerr = abs((double)omega - omegad) < 1.e-10 ? -99. : 10.*log10(abs((double)omega - omegad) / omegad);
    if (merr > merrmax) { merrmax = merr; }
    if (oerr > oerrmax) { oerrmax = oerr; }
  }
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("naive  float  ");
  printf(fmt, duration, mu, merr, omega, oerr);
  
  /*
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    yc_f(&omega, &mu, in);
  }
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  merr = abs((double)mu - mud) < 1.e-7 ? -99. : 10.*log10( abs(( (double)mu - mud) / mud) );
  oerr = abs((double)omega - omegad) < 1.e-7 ? -99. : 10.*log10(abs((double)omega - omegad) / omegad);
  printf("YC            ");
  printf(fmt, duration, mu, merr, omega, oerr);

  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    yc2_f(&omega, &mu, in);
  }
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  merr = abs((double)mu - mud) < 1.e-7 ? -99. : 10.*log10( abs(( (double)mu - mud) / mud) );
  oerr = abs((double)omega - omegad) < 1.e-7 ? -99. : 10.*log10(abs((double)omega - omegad) / omegad);
  printf("YC2           ");
  printf(fmt, duration, mu, merr, omega, oerr);
*/

/*
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    yc4_f(&omega, &mu, in);
  }
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  oerr = abs(omega - (float)omegad) / omegad;
  merr = abs(mu - (float)mud) / mud;
  printf("naive  float  %9.2f  %12.7f   %5.2e  %12.7f     %5.2e\n", duration, mu, merr, omega, oerr);

  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    generic_f(&omega, &mu, in);
  }
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  oerr = abs(omega - (float)omegad) / omegad;
  merr = abs(mu - (float)mud) / mud;
  printf("naive  float  %9.2f  %12.7f   %5.2e  %12.7f     %5.2e\n", duration, mu, merr, omega, oerr);
*/

  merrmax = oerrmax = -200;
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    volk_32f_stddev_and_mean_32f_x2_manual( &omega, &mu, in, N, "generic");
    merr = abs((double)mu - mud) < 1.e-10 ? -99. : 10.*log10( abs(( (double)mu - mud) / mud) );
    oerr = abs((double)omega - omegad) < 1.e-10 ? -99. : 10.*log10(abs((double)omega - omegad) / omegad);
    if (merr > merrmax) { merrmax = merr; }
    if (oerr > oerrmax) { oerrmax = oerr; }  
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("VOLK generic  ");
  printf(fmt, duration, mu, merr, omega, oerr);


  merrmax = oerrmax = -200;
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    volk_32f_stddev_and_mean_32f_x2_manual( &omega, &mu, in, N, "u_sse");
    merr = abs((double)mu - mud) < 1.e-10 ? -99. : 10.*log10( abs(( (double)mu - mud) / mud) );
    oerr = abs((double)omega - omegad) < 1.e-10 ? -99. : 10.*log10(abs((double)omega - omegad) / omegad);
    if (merr > merrmax) { merrmax = merr; }
    if (oerr > oerrmax) { oerrmax = oerr; }    
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("VOLK u_sse    ");
  printf(fmt, duration, mu, merr, omega, oerr);

  merrmax = oerrmax = -200;  
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    volk_32f_stddev_and_mean_32f_x2_manual( &omega, &mu, in, N, "a_sse");
    merr = abs((double)mu - mud) < 1.e-10 ? -99. : 10.*log10( abs(( (double)mu - mud) / mud) );
    oerr = abs((double)omega - omegad) < 1.e-10 ? -99. : 10.*log10(abs((double)omega - omegad) / omegad);
    if (merr > merrmax) { merrmax = merr; }
    if (oerr > oerrmax) { oerrmax = oerr; }    
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("VOLK a_sse    ");
  printf(fmt, duration, mu, merr, omega, oerr);

  merrmax = oerrmax = -200;
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    volk_32f_stddev_and_mean_32f_x2_manual( &omega, &mu, in, N, "u_avx");
    merr = abs((double)mu - mud) < 1.e-10 ? -99. : 10.*log10( abs(( (double)mu - mud) / mud) );
    oerr = abs((double)omega - omegad) < 1.e-10 ? -99. : 10.*log10(abs((double)omega - omegad) / omegad);
    if (merr > merrmax) { merrmax = merr; }
    if (oerr > oerrmax) { oerrmax = oerr; }    
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("VOLK u_avx    ");
  printf(fmt, duration, mu, merr, omega, oerr);

  merrmax = oerrmax = -200;
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    volk_32f_stddev_and_mean_32f_x2_manual( &omega, &mu, in, N, "a_avx");
    merr = abs((double)mu - mud) < 1.e-10 ? -99. : 10.*log10( abs(( (double)mu - mud) / mud) );
    oerr = abs((double)omega - omegad) < 1.e-10 ? -99. : 10.*log10(abs((double)omega - omegad) / omegad);
    if (merr > merrmax) { merrmax = merr; }
    if (oerr > oerrmax) { oerrmax = oerr; }    
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("VOLK a_avx    ");
  printf(fmt, duration, mu, merr, omega, oerr);

/*
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    //volk_std(&omega, &mu, in);
    volk_32f_stddev_and_mean_32f_x2_manual( &omega, &mu, &in[0], in.size(), "u_fma");
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("VOLK U_FMA   %9.2f %12.7f %12.7f\n", duration, mu, omega);

  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    //volk_std(&omega, &mu, in);
    volk_32f_s32f_stddev_32f_manual( &omega, &in[0], mu, in.size(), "u_avx");
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("VOLK STDDEV  %9.2f %12.7f %12.7f\n", duration, mu, omega);  

  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    volk2_std(&omega, &mu, in);
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  cout << "volk_std took " << duration << " us" << endl;
  cout << "mu   :" <<  mu << endl;
  cout << "omega:" <<  omega << endl << endl;
  
  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    volk2_std(&omega, &best_mu, in,true);
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  cout << "volk_std with supplied mean took " << duration << " us" << endl;
  cout << "mu   :" <<  mu << endl;
  cout << "omega:" <<  omega << endl << endl;

  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    welford_stdmean_f(&omega, &mu, in);
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("WELFORD      %9.2f %12.7f %12.7f\n", duration, mu, omega);
 

  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    youngscramer_stdmean_f(&omega, &mu, in);
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("YC           %9.2f %12.7f %12.7f\n", duration, mu, omega);

  */
  /*

  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    youngscramerdiv2_stdmean_f(&omega, &mu, in);
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("YC DIV 2     %9.2f %12.7f %12.7f\n", duration, mu, omega);

  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    youngscramerdiv4_stdmean_f(&omega, &mu, in);
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("YC DIV 4     %9.2f %12.7f %12.7f\n", duration, mu, omega);
 

  t1 = high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
  {
    youngscramerdiv8_stdmean_f(&omega, &mu, in);
  }  
  t2 = high_resolution_clock::now();
  duration = duration_cast<microseconds>( t2 - t1 ).count()/( (double) iters);
  printf("YC DIV 8     %9.2f %12.7f %12.7f\n", duration, mu, omega);

  */

  return 0;
}