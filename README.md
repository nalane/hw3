# hw3
This is my submission for homework assignment 3 in CSCI 550: Advanced Data Mining. The instructions for use are in the PDF I submitted to Brightspace, but in brief, this project is programmed in Rust and uses Rust's build system, Cargo. Cargo may be installed using the instructions at this webpage: https://doc.rust-lang.org/cargo/getting-started/installation.html.

Once Cargo is installed, the program may be run with

  `cargo run`
  
To see the list of command line arguments this program takes, you may pass it the -h flag:

  `cargo run -- -h`
  
Finally, I recommend using an optimized build, as it runs orders of magnitude faster than the unoptimized build:

  `cargo run --release -- -h`
