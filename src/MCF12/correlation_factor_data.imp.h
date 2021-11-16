#ifdef HAVE_CUDA
class FUNCTIONAL_NAME(_Correlation_Factor_Data_Device) :
    public Correlation_Factor_Data<thrust::device_vector, thrust::device_allocator> {
 public:
  FUNCTIONAL_NAME(_Correlation_Factor_Data_Device)(int electrons_in,
      int electron_pairs_in,
      double gamma_in,
      double beta_in);
  void update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) override;
};
#endif
