use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct MeshSummary {
    pub n_vertices: usize,
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
}

pub fn read_mesh_summary<P: AsRef<Path>>(path: P) -> Result<MeshSummary, String> {
    let content = fs::read_to_string(path.as_ref()).map_err(|e| {
        format!(
            "failed to read mesh file '{}': {e}",
            path.as_ref().display()
        )
    })?;

    let mut n_vertices = 0usize;
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut fields = line.split_whitespace();
        let x = match fields.next().and_then(|v| v.parse::<f64>().ok()) {
            Some(v) => v,
            None => continue,
        };
        let y = match fields.next().and_then(|v| v.parse::<f64>().ok()) {
            Some(v) => v,
            None => continue,
        };

        n_vertices += 1;
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }

    if n_vertices == 0 {
        return Err("mesh file did not contain any parseable 'x y' vertex lines".to_string());
    }

    Ok(MeshSummary {
        n_vertices,
        xmin,
        xmax,
        ymin,
        ymax,
    })
}
