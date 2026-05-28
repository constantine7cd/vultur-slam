#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose3 {
    column_major: [f64; 16],
}

impl Pose3 {
    pub fn identity() -> Self {
        Self {
            column_major: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    pub fn translated(x: f64, y: f64, z: f64) -> Self {
        let mut pose = Self::identity();
        pose.column_major[12] = x;
        pose.column_major[13] = y;
        pose.column_major[14] = z;
        pose
    }

    pub fn translation(self) -> [f64; 3] {
        [
            self.column_major[12],
            self.column_major[13],
            self.column_major[14],
        ]
    }

    pub fn column_major(self) -> [f64; 16] {
        self.column_major
    }

    pub fn with_translation(mut self, translation: [f64; 3]) -> Self {
        self.column_major[12] = translation[0];
        self.column_major[13] = translation[1];
        self.column_major[14] = translation[2];
        self
    }
}
