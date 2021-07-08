mod ml;
use ml::{relu, sigmoid};

fn main() {
    println!("sigmoid of -1: {:?}", sigmoid(-1.0));
    println!("relu of -1: {:?}", relu(-1.0));
}
