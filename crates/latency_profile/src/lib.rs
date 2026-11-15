//! Shared, validated mean-call latency profiles. All times are in seconds.
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

pub type Curve = BTreeMap<u32, f32>;
pub type LatencyData = HashMap<String, HashMap<String, Curve>>;

/// Parse the collector's simple, unquoted CSV format by column name.
pub fn parse_csv(text: &str) -> Result<LatencyData, String> {
    let mut lines = text.lines();
    let header: Vec<_> = lines
        .next()
        .ok_or("empty latency CSV")?
        .split(',')
        .map(str::trim)
        .collect();
    let column = |names: &[&str]| {
        header
            .iter()
            .position(|h| names.contains(h))
            .ok_or_else(|| format!("missing CSV column {}", names[0]))
    };
    let model = column(&["model_name"])?;
    let profile = column(&["split_param", "config_name"])?;
    let batch = column(&["batch_size"])?;
    let latency = column(&["avg_latency(s)", "latency_in_seconds"])?;
    let mut data = LatencyData::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        let fail = |msg: &str| format!("CSV line {}: {msg}", i + 2);
        if fields.len() != header.len() || line.contains('"') {
            return Err(fail("invalid unquoted CSV row"));
        }
        let b: u32 = fields[batch]
            .parse()
            .map_err(|_| fail("invalid batch size"))?;
        let l: f32 = fields[latency]
            .parse()
            .map_err(|_| fail("invalid latency in seconds"))?;
        if b == 0
            || !l.is_finite()
            || l <= 0.0
            || fields[model].is_empty()
            || fields[profile].is_empty()
        {
            return Err(fail(
                "names, batch and finite latency must be nonempty/positive",
            ));
        }
        let curve = data
            .entry(fields[model].to_ascii_lowercase())
            .or_default()
            .entry(fields[profile].to_ascii_lowercase())
            .or_default();
        if curve.insert(b, l).is_some() {
            return Err(fail("duplicate model/profile/batch"));
        }
    }
    if data.is_empty() {
        return Err("latency CSV has no records".into());
    }
    Ok(data)
}

pub fn load_csv(path: &Path) -> Result<LatencyData, String> {
    parse_csv(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?)
}

/// Exact points retain measured values, including decreasing segments. No extrapolation.
pub fn interpolate(curve: &Curve, batch: u32) -> Result<f32, String> {
    if let Some(&latency) = curve.get(&batch) {
        return Ok(latency);
    }
    match (
        curve.range(..batch).next_back(),
        curve.range(batch..).next(),
    ) {
        (Some((&a, &x)), Some((&b, &y))) => Ok(x + (batch - a) as f32 * (y - x) / (b - a) as f32),
        _ => Err(format!(
            "batch {batch} outside measured range {:?}",
            curve
                .first_key_value()
                .zip(curve.last_key_value())
                .map(|((a, _), (b, _))| (*a, *b))
        )),
    }
}

pub fn curve<'a>(data: &'a LatencyData, model: &str, profile: &str) -> Result<&'a Curve, String> {
    data.get(&model.to_ascii_lowercase())
        .and_then(|p| p.get(&profile.to_ascii_lowercase()))
        .ok_or_else(|| format!("missing latency curve {model}/{profile}"))
}

/// Aliases are resolved against the original table, never recursively or by suffix matching.
pub fn apply_aliases(data: &mut LatencyData, aliases: &HashMap<String, String>) {
    for profiles in data.values_mut() {
        let extra: Vec<_> = aliases
            .iter()
            .filter_map(|(name, key)| {
                profiles
                    .get(&key.to_ascii_lowercase())
                    .map(|c| (name.to_ascii_lowercase(), c.clone()))
            })
            .collect();
        profiles.extend(extra);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const HEADER: &str = "model_name,split_param,batch_size,avg_latency(s),avg_latency\n";
    #[test]
    fn measured_and_decreasing_interpolation() {
        let d = parse_csv(&format!("{HEADER}m,p,2,0.04,40000\nm,p,4,0.02,20000\n")).unwrap();
        let c = curve(&d, "M", "p").unwrap();
        assert_eq!(interpolate(c, 2).unwrap(), 0.04);
        assert!((interpolate(c, 3).unwrap() - 0.03).abs() < 1e-7);
        assert!(interpolate(c, 0).is_err());
        assert!(interpolate(c, 5).is_err());
    }
    #[test]
    fn rejects_invalid_rows() {
        for rows in [
            "m,p,0,1,1\n",
            "m,p,1,NaN,1\n",
            "m,p,1,-1,1\n",
            "m,p,1,1,1\nm,p,1,2,2\n",
            "m,p,1,1\n",
        ] {
            assert!(parse_csv(&format!("{HEADER}{rows}")).is_err());
        }
    }
    #[test]
    fn original_measurements_and_explicit_aliases() {
        let text = include_str!("../../../logs/model_latency.csv");
        let mut d = parse_csv(text).unwrap();
        assert_eq!(
            d.values()
                .flat_map(|p| p.values())
                .map(Curve::len)
                .sum::<usize>(),
            1760
        );
        for line in text.lines().skip(1) {
            let f: Vec<_> = line.split(',').collect();
            assert_eq!(
                interpolate(curve(&d, f[0], f[1]).unwrap(), f[2].parse().unwrap()).unwrap(),
                f[3].parse::<f32>().unwrap()
            );
        }
        apply_aliases(
            &mut d,
            &HashMap::from([("vir05_1c_16g".into(), "vir05_1c_8g".into())]),
        );
        assert_eq!(
            curve(&d, "resnet50", "vir05_1c_16g").unwrap(),
            curve(&d, "resnet50", "vir05_1c_8g").unwrap()
        );
        assert!(curve(&d, "unknown", "1g.10gb").is_err());
    }
}
