//! Integrates local application settings.
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::{Error, Result};

const NO_PROXY_KEY: &str = "http.noProxy";
const KEYS: [&str; 5] = [
    "http.proxy",
    "http.proxyKerberosServicePrincipal",
    "http.proxySupport",
    "cursor.general.disableHttp2",
    "http.experimental.systemCertificatesV2",
];

const MACHINE_USE_LOCAL_PROXY_KEY: &str = "http.useLocalProxyConfiguration";
const MACHINE_MANAGED_KEYS: [&str; 3] = [
    MACHINE_USE_LOCAL_PROXY_KEY,
    "http.proxy",
    "http.proxySupport",
];

fn resolve_home() -> Result<PathBuf> {
    dirs::home_dir().ok_or_else(|| Error::Config("cannot resolve user home directory".into()))
}

fn uses_real_home(home: &Path) -> bool {
    dirs::home_dir().as_deref() == Some(home)
}

fn desktop_settings_path(home: &Path) -> Result<PathBuf> {
    match std::env::consts::OS {
        "macos" => Ok(home.join("Library/Application Support/Cursor/User/settings.json")),
        "windows" => {
            let root = if uses_real_home(home) {
                std::env::var_os("APPDATA")
                    .filter(|v| !v.is_empty())
                    .map(PathBuf::from)
                    .unwrap_or_else(|| home.join("AppData/Roaming"))
            } else {
                home.join("AppData/Roaming")
            };
            Ok(root.join("Cursor/User/settings.json"))
        }
        "linux" => {
            let root = if uses_real_home(home) {
                std::env::var_os("XDG_CONFIG_HOME")
                    .filter(|v| !v.is_empty())
                    .map(PathBuf::from)
                    .unwrap_or_else(|| home.join(".config"))
            } else {
                home.join(".config")
            };
            Ok(root.join("Cursor/User/settings.json"))
        }
        platform => Err(Error::Config(format!(
            "Cursor settings are unsupported on {platform}"
        ))),
    }
}

fn cursor_server_present(home: &Path) -> bool {
    home.join(".cursor-server").is_dir()
}

fn machine_settings_path(home: &Path) -> PathBuf {
    home.join(".cursor-server/data/Machine/settings.json")
}

fn path() -> Result<PathBuf> {
    desktop_settings_path(&resolve_home()?)
}

fn read_at(path: &Path) -> Result<BTreeMap<String, Value>> {
    let data = match fs::read_to_string(path) {
        Ok(data) => data,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(error) => return Err(error.into()),
    };
    if data.trim().is_empty() {
        return Ok(BTreeMap::new());
    }
    json5::from_str(&data)
        .map_err(|error| Error::Config(format!("parse Cursor settings JSONC: {error}")))
}

fn read() -> Result<BTreeMap<String, Value>> {
    read_at(&path()?)
}

fn write_at(path: &Path, settings: &BTreeMap<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let data = serde_json::to_vec_pretty(settings)?;
    let temp = path.with_extension("json.tmp");
    fs::write(&temp, [data.as_slice(), b"\n"].concat())?;
    fs::rename(temp, path)?;
    Ok(())
}

fn write_machine_at(path: &Path, settings: &BTreeMap<String, Value>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = [serde_json::to_vec_pretty(settings)?.as_slice(), b"\n"].concat();
    let temp = path.with_extension("json.tmp");
    fs::write(&temp, &payload)?;
    match fs::rename(&temp, path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::CrossesDevices => {
            fs::write(path, &payload)?;
            let _ = fs::remove_file(&temp);
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

fn apply_desktop_proxy(settings: &mut BTreeMap<String, Value>, proxy_url: &str) {
    settings.remove(NO_PROXY_KEY);
    settings.insert(KEYS[0].into(), Value::String(proxy_url.into()));
    settings.insert(KEYS[1].into(), Value::String(proxy_url.into()));
    settings.insert(KEYS[2].into(), Value::String("on".into()));
    settings.insert(KEYS[3].into(), Value::Bool(true));
    settings.insert(KEYS[4].into(), Value::Bool(true));
}

fn apply_machine_proxy(settings: &mut BTreeMap<String, Value>, proxy_url: &str) {
    settings.remove(NO_PROXY_KEY);
    settings.insert(MACHINE_USE_LOCAL_PROXY_KEY.into(), Value::Bool(false));
    settings.insert(KEYS[0].into(), Value::String(proxy_url.into()));
    settings.insert(KEYS[2].into(), Value::String("override".into()));
}

fn desktop_settings_match(settings: &BTreeMap<String, Value>, proxy_url: &str) -> bool {
    settings.get(KEYS[0]) == Some(&Value::String(proxy_url.into()))
        && settings.get(KEYS[1]) == Some(&Value::String(proxy_url.into()))
        && settings.get(KEYS[2]) == Some(&Value::String("on".into()))
        && settings.get(KEYS[3]) == Some(&Value::Bool(true))
        && settings.get(KEYS[4]) == Some(&Value::Bool(true))
}

fn machine_settings_match(settings: &BTreeMap<String, Value>, proxy_url: &str) -> bool {
    settings.get(MACHINE_USE_LOCAL_PROXY_KEY) == Some(&Value::Bool(false))
        && settings.get(KEYS[0]) == Some(&Value::String(proxy_url.into()))
        && settings.get(KEYS[2]) == Some(&Value::String("override".into()))
}

fn write_proxy_settings_at(home: &Path, proxy_url: &str) -> Result<()> {
    let desktop_path = desktop_settings_path(home)?;
    let mut desktop = read_at(&desktop_path)?;
    apply_desktop_proxy(&mut desktop, proxy_url);
    write_at(&desktop_path, &desktop)?;

    if cursor_server_present(home) {
        let machine_path = machine_settings_path(home);
        let mut machine = read_at(&machine_path)?;
        apply_machine_proxy(&mut machine, proxy_url);
        write_machine_at(&machine_path, &machine)?;
    }
    Ok(())
}

fn clear_proxy_settings_at(home: &Path) -> Result<()> {
    let desktop_path = desktop_settings_path(home)?;
    let mut desktop = read_at(&desktop_path)?;
    let before = desktop.len();
    for key in KEYS {
        desktop.remove(key);
    }
    if desktop.len() != before {
        write_at(&desktop_path, &desktop)?;
    }

    let machine_path = machine_settings_path(home);
    if machine_path.is_file() {
        let mut machine = read_at(&machine_path)?;
        let machine_before = machine.len();
        for key in MACHINE_MANAGED_KEYS {
            machine.remove(key);
        }
        if machine.len() != machine_before {
            write_machine_at(&machine_path, &machine)?;
        }
    }
    Ok(())
}

fn settings_match_at(home: &Path, proxy_url: &str) -> Result<bool> {
    let desktop_path = desktop_settings_path(home)?;
    let desktop = read_at(&desktop_path)?;
    if !desktop_settings_match(&desktop, proxy_url) {
        return Ok(false);
    }
    if cursor_server_present(home) {
        let machine_path = machine_settings_path(home);
        let machine = read_at(&machine_path)?;
        return Ok(machine_settings_match(&machine, proxy_url));
    }
    Ok(true)
}

pub fn write_proxy_settings(proxy_url: &str) -> Result<()> {
    write_proxy_settings_at(&resolve_home()?, proxy_url)
}

pub fn clear_proxy_settings() -> Result<()> {
    clear_proxy_settings_at(&resolve_home()?)
}

pub fn settings_match(proxy_url: &str) -> Result<bool> {
    settings_match_at(&resolve_home()?, proxy_url)
}

pub fn clear_stale_managed_settings() -> Result<()> {
    let settings = read()?;
    let managed_signature = settings.get(KEYS[2]) == Some(&Value::String("on".into()))
        && settings.get(KEYS[3]) == Some(&Value::Bool(true))
        && settings.get(KEYS[4]) == Some(&Value::Bool(true));
    let loopback = settings
        .get(KEYS[0])
        .and_then(Value::as_str)
        .and_then(|value| value.parse::<reqwest::Url>().ok())
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| matches!(host.as_str(), "127.0.0.1" | "localhost" | "::1"));
    if managed_signature && loopback {
        clear_proxy_settings()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    const PROXY: &str = "http://127.0.0.1:18789";

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    use std::sync::Mutex;

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn without_cursor_server_does_not_create_it() {
        let home = tempfile::tempdir().unwrap();
        write_proxy_settings_at(home.path(), PROXY).unwrap();
        assert!(!home.path().join(".cursor-server").exists());
        assert!(desktop_settings_path(home.path()).unwrap().is_file());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn custom_home_stays_under_home_when_xdg_config_home_set() {
        let _guard = ENV_LOCK.lock().unwrap();
        let home = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("XDG_CONFIG_HOME");
        // SAFETY: ENV_LOCK serializes mutation of process environment variables.
        unsafe { std::env::set_var("XDG_CONFIG_HOME", outside.path()) };

        write_proxy_settings_at(home.path(), PROXY).unwrap();

        let desktop = desktop_settings_path(home.path()).unwrap();
        assert!(desktop.starts_with(home.path()));
        assert!(desktop.is_file());
        assert!(!outside.path().join("Cursor").exists());

        match previous {
            Some(value) => unsafe { std::env::set_var("XDG_CONFIG_HOME", value) },
            None => unsafe { std::env::remove_var("XDG_CONFIG_HOME") },
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn custom_home_stays_under_home_when_appdata_set() {
        let _guard = ENV_LOCK.lock().unwrap();
        let home = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("APPDATA");
        // SAFETY: ENV_LOCK serializes mutation of process environment variables.
        unsafe { std::env::set_var("APPDATA", outside.path()) };

        write_proxy_settings_at(home.path(), PROXY).unwrap();

        let desktop = desktop_settings_path(home.path()).unwrap();
        assert!(desktop.starts_with(home.path()));
        assert!(desktop.is_file());
        assert!(!outside.path().join("Cursor").exists());

        match previous {
            Some(value) => unsafe { std::env::set_var("APPDATA", value) },
            None => unsafe { std::env::remove_var("APPDATA") },
        }
    }

    #[test]
    fn with_cursor_server_writes_machine_settings() {
        let home = tempfile::tempdir().unwrap();
        let machine_dir = home.path().join(".cursor-server/data/Machine");
        fs::create_dir_all(&machine_dir).unwrap();
        let machine_path = machine_dir.join("settings.json");
        fs::write(
            &machine_path,
            r#"{
  // keep me
  "editor.fontSize": 14,
  "http.noProxy": ["localhost"]
}"#,
        )
        .unwrap();

        write_proxy_settings_at(home.path(), PROXY).unwrap();

        let machine: BTreeMap<String, Value> =
            json5::from_str(&fs::read_to_string(&machine_path).unwrap()).unwrap();
        assert_eq!(machine.get("editor.fontSize"), Some(&json!(14)));
        assert!(!machine.contains_key(NO_PROXY_KEY));
        assert_eq!(
            machine.get(MACHINE_USE_LOCAL_PROXY_KEY),
            Some(&Value::Bool(false))
        );
        assert_eq!(machine.get(KEYS[0]), Some(&Value::String(PROXY.into())));
        assert_eq!(
            machine.get(KEYS[2]),
            Some(&Value::String("override".into()))
        );
    }

    #[test]
    fn settings_match_requires_machine_settings_when_cursor_server_exists() {
        let home = tempfile::tempdir().unwrap();
        fs::create_dir_all(home.path().join(".cursor-server")).unwrap();
        write_proxy_settings_at(home.path(), PROXY).unwrap();
        assert!(settings_match_at(home.path(), PROXY).unwrap());

        let machine_path = machine_settings_path(home.path());
        fs::remove_file(machine_path).unwrap();
        assert!(!settings_match_at(home.path(), PROXY).unwrap());

        write_proxy_settings_at(home.path(), PROXY).unwrap();
        assert!(settings_match_at(home.path(), PROXY).unwrap());

        let mut machine = read_at(&machine_settings_path(home.path())).unwrap();
        machine.insert(KEYS[0].into(), Value::String("http://127.0.0.1:1".into()));
        write_machine_at(&machine_settings_path(home.path()), &machine).unwrap();
        assert!(!settings_match_at(home.path(), PROXY).unwrap());
    }

    #[test]
    fn clear_proxy_settings_keeps_unrelated_machine_keys() {
        let home = tempfile::tempdir().unwrap();
        let machine_dir = home.path().join(".cursor-server/data/Machine");
        fs::create_dir_all(&machine_dir).unwrap();
        let machine_path = machine_dir.join("settings.json");
        fs::write(
            &machine_path,
            r#"{"editor.fontSize": 16, "http.proxy": "http://127.0.0.1:18789"}"#,
        )
        .unwrap();
        write_proxy_settings_at(home.path(), PROXY).unwrap();

        clear_proxy_settings_at(home.path()).unwrap();

        let machine: BTreeMap<String, Value> =
            json5::from_str(&fs::read_to_string(&machine_path).unwrap()).unwrap();
        assert_eq!(machine.get("editor.fontSize"), Some(&json!(16)));
        for key in MACHINE_MANAGED_KEYS {
            assert!(!machine.contains_key(key));
        }
    }
}
