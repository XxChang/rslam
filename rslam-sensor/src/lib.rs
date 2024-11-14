pub mod pinhole_camera;

#[cfg(test)]
mod tests {
    use std::io::BufRead;

    #[test]
    fn test_pinhole_camera() {
        let calib_file_path = std::path::Path::new("../datasets/01/calib.txt");

        let file = std::fs::File::open(&calib_file_path).unwrap();
        let file = std::io::BufReader::new(file);

        let mut lines = file.lines().filter_map(|l| l.ok());
        println!("test pinhole camera");
        while let Some(l) = lines.next() {
            println!("{l:?}");
        }
    }
}
