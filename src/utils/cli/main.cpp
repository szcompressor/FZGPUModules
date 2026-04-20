#include <exception>
#include <iostream>

int fzgmod_cli_main(int argc, char** argv);

int main(int argc, char** argv) {
    try {
        return fzgmod_cli_main(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[fzgmod-cli] fatal: " << e.what() << "\n";
        return 1;
    }
}
