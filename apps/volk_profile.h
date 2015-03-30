

void read_results(std::vector<volk_test_results_t> *results);
void write_results(const std::vector<volk_test_results_t> *results, bool update_result);
void write_json(std::ofstream &json_file, std::vector<volk_test_results_t> results);
