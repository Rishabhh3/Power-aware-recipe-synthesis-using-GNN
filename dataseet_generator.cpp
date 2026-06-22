#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cstdlib>
#include <random>
#include <filesystem>
#include <fstream>

using namespace std;
namespace fs = filesystem;

// --- Configuration ---
const string VERILOG_DIR = "EPFL Circuits/arithmetic"; // Updated to actual dir for MacOS compatibility
const string OUTPUT_BENCH_DIR = "dataset/bench_files";
const string OUTPUT_LOG_DIR = "dataset/abc_logs";
const string TECH_LIB_PATH = "abc/NangateOpenCellLibrary_typical.lib";

const string ABC_BINARY = "./abc/abc"; // Updated for Mac

const int RECIPE_LENGTH = 20;
const int RECIPES_PER_CIRCUIT = 1000;
const int NUM_THREADS = thread::hardware_concurrency(); // Use all available CPU cores

const vector<string> COMMAND_POOL = {
    "rewrite", "rewrite -z", "refactor", "refactor -z", 
    "resub", "resub -z", "balance"
};

// --- Helper: Generate Random Recipe ---
vector<string> generate_recipe() {
    thread_local mt19937 generator(random_device{}());
    uniform_int_distribution<int> distribution(0, COMMAND_POOL.size() - 1);
    
    vector<string> recipe;
    for (int i = 0; i < RECIPE_LENGTH; ++i) {
        recipe.push_back(COMMAND_POOL[distribution(generator)]);
    }
    return recipe;
}

// --- Worker Thread Function ---
// Simplified by giving each thread a strided subset of work, avoiding complex mutexes and jobs queues!
void worker_thread(int thread_id, const vector<fs::path>& verilog_files) {
    for (const auto& v_file : verilog_files) {
        string circuit_name = v_file.stem().string();
        string verilog_path = v_file.string();

        // Stride loops: each thread takes every 'NUM_THREADS'-th run
        for (int run_id = thread_id; run_id < RECIPES_PER_CIRCUIT; run_id += NUM_THREADS) {
            string log_path = OUTPUT_LOG_DIR + "/" + circuit_name + "_run" + to_string(run_id) + ".log";
            
            // If the log file already exists and isn't empty, skip this job
            if (fs::exists(log_path) && fs::file_size(log_path) > 0) {
                continue; 
            }

            vector<string> recipe = generate_recipe();
            string abc_script = "read " + verilog_path + "; strash; ";
            string recipe_log_str = "RECIPE: ";
            
            for (int step = 0; step < RECIPE_LENGTH; ++step) {
                string step_bench_path = OUTPUT_BENCH_DIR + "/" + 
                                         circuit_name + "_run" + to_string(run_id) + 
                                         "_step" + to_string(step) + ".bench";
                                              
                abc_script += recipe[step] + "; write_bench " + step_bench_path + "; ";
                recipe_log_str += recipe[step] + "; ";
            }
            
            abc_script += "read_lib " + TECH_LIB_PATH + "; map; print_stats -p";
            
            // Write the recipe string to the log file first
            ofstream log_file(log_path);
            if (log_file.is_open()) {
                log_file << recipe_log_str << "\n";
                log_file.close();
            }

            // Redirect ABC's stdout and stderr to append to our log file
            string shell_command = ABC_BINARY + " -c \"" + abc_script + "\" >> " + log_path + " 2>&1";
            
            // Execute the system call
            int result = system(shell_command.c_str());
            
            if (result != 0) {
                cerr << "Thread " << thread_id << " failed on job: " << circuit_name 
                     << " run " << run_id << endl;
            }
        }
    }
}

int main() {
    // Setup directories
    fs::create_directories(OUTPUT_BENCH_DIR);
    fs::create_directories(OUTPUT_LOG_DIR);

    // Read all files
    vector<fs::path> verilog_files;
    string target_dir = VERILOG_DIR;

    if (fs::exists(target_dir)) {
        for (const auto& entry : fs::directory_iterator(target_dir)) {
            if (entry.path().extension() == ".v") { // Filtering explicitly for .v files
                verilog_files.push_back(entry.path());
            }
        }
    } else {
        cerr << "Verilog directory not found at " << target_dir << endl;
        return 1;
    }

    cout << "Found " << verilog_files.size() << " Verilog files.\n";
    cout << "Starting " << NUM_THREADS << " worker threads...\n";

    // Launch worker threads with stride pattern
    vector<thread> workers;
    for (int i = 0; i < NUM_THREADS; ++i) {
        workers.emplace_back(worker_thread, i, ref(verilog_files));
    }

    // Wait for all threads to finish
    for (auto& worker : workers) {
        worker.join();
    }

    cout << "Data generation successfully completed.\n";
    return 0;
}