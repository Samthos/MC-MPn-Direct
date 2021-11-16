#ifdef HAVE_CUDA
__global__ void FUNCTIONAL_NAME(_f12p_kernal)(
    double gamma, double beta,
    int size,
    const double* r12, double* f12p) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    f12p[tid] = FUNCTIONAL_NAME(_f12)(r12[tid], gamma, beta);
  }
}

__global__ void FUNCTIONAL_NAME(_f12p_a_kernal)(
    double gamma, double beta, int size,
    const double* r12, double* f12p_a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    f12p_a[tid] = FUNCTIONAL_NAME(_f12_a)(r12[tid], gamma, beta);
  }
}

__global__ void FUNCTIONAL_NAME(_f12p_c_kernal)(
    double gamma, double beta,
    int size, const double* r12, double* f12p_c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    f12p_c[tid] = FUNCTIONAL_NAME(_f12_c)(r12[tid], gamma, beta);
  }
}

__global__ void FUNCTIONAL_NAME(_f12o_kernal)(
    double gamma, double beta,
    int size, const Point* pos, double* f12o) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = tidy * size + tidx;
  if (tidx < size && tidy < size && tidx != tidy) {
    auto dr = Point::distance(pos[tidx], pos[tidy]);
    f12o[tid] = FUNCTIONAL_NAME(_f12)(dr, gamma, beta);
  }
}

__global__ void FUNCTIONAL_NAME(_f12o_b_kernal)(
    double gamma, double beta,
    int size, const Point* pos, double* f12o_b) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = tidy * size + tidx;
  if (tidx < size && tidy < size && tidx != tidy) {
    auto dr = Point::distance(pos[tidx], pos[tidy]);
    f12o_b[tid] = FUNCTIONAL_NAME(_f12_b)(dr, gamma, beta);
  }
}

__global__ void FUNCTIONAL_NAME(_f12o_d_kernal)(
    double gamma, double beta,
    int size, const Point* pos, double* f12o_d) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = tidy * size + tidx;
  if (tidx < size && tidy < size && tidx != tidy) {
    auto dr = Point::distance(pos[tidx], pos[tidy]);
    f12o_d[tid] = FUNCTIONAL_NAME(_f12_d)(dr, gamma, beta);
  }
}

__global__ void FUNCTIONAL_NAME(_f13_kernal)(
    double gamma, double beta,
    int electron_pairs, const Point* electron_pair_pos,
    int electrons, const Point* electron_pos, double* f13) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = tidy * electrons + tidx;
  if (tidx < electrons && tidy < electron_pairs) {
    auto dr = Point::distance(electron_pair_pos[tidy], electron_pos[tidx]);
    f13[tid] = FUNCTIONAL_NAME(_f12)(dr, gamma, beta);
  }
}

FUNCTIONAL_NAME(_Correlation_Factor_Data_Device)::FUNCTIONAL_NAME(_Correlation_Factor_Data_Device)(int electrons_in,
    int electron_pairs_in,
    double gamma_in,
    double beta_in) : 
  Correlation_Factor_Data(electrons_in, 
      electron_pairs_in,
      CORRELATION_FACTOR::FUNCTIONAL_NAME(),
      gamma_in,
      beta_in) {};

void FUNCTIONAL_NAME(_Correlation_Factor_Data_Device)::update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  dim3 block_size(128, 1, 1);
  dim3 grid_size((electron_pair_list->size() + 127) / 128, 1, 1);
  FUNCTIONAL_NAME(_f12p_kernal)<<<grid_size, block_size>>>(gamma, beta, electron_pair_list->size(), electron_pair_list->r12.data().get(), f12p.data().get());
  FUNCTIONAL_NAME(_f12p_a_kernal)<<<grid_size, block_size>>>(gamma, beta, electron_pair_list->size(), electron_pair_list->r12.data().get(), f12p_a.data().get());
  FUNCTIONAL_NAME(_f12p_c_kernal)<<<grid_size, block_size>>>(gamma, beta, electron_pair_list->size(), electron_pair_list->r12.data().get(), f12p_c.data().get());

  block_size = dim3(16, 16, 1);
  grid_size = dim3((electron_list->size() + 15) / 16, (electron_list->size() + 15) / 16, 1);
  FUNCTIONAL_NAME(_f12o_kernal)<<<grid_size, block_size>>>(gamma, beta, electron_list->size(), electron_list->pos.data().get(), f12o.data().get());
  FUNCTIONAL_NAME(_f12o_b_kernal)<<<grid_size, block_size>>>(gamma, beta, electron_list->size(), electron_list->pos.data().get(), f12o_b.data().get());
  FUNCTIONAL_NAME(_f12o_d_kernal)<<<grid_size, block_size>>>(gamma, beta, electron_list->size(), electron_list->pos.data().get(), f12o_d.data().get());

  block_size = dim3(16, 16, 1);
  grid_size = dim3((electron_list->size() + 15) / 16, (electron_pair_list->size() + 15) / 16, 1);
  FUNCTIONAL_NAME(_f13_kernal)<<<grid_size, block_size>>>(gamma, beta,
      electron_pair_list->size(), electron_pair_list->pos1.data().get(),
      electron_list->size(), electron_list->pos.data().get(),
      f13.data().get());
  FUNCTIONAL_NAME(_f13_kernal)<<<grid_size, block_size>>>(gamma, beta,
      electron_pair_list->size(), electron_pair_list->pos2.data().get(),
      electron_list->size(), electron_list->pos.data().get(),
      f23.data().get());
}
#endif
