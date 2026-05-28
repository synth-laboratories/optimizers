use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::config::CacheConfigMode;
use crate::error::{OptimizerError, Result};

const CACHE_SCHEMA_VERSION: &str = "synth_optimizers.request_response_cache.v1";
const DEFAULT_REQUEST_CACHE_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CacheMode {
    Off,
    Readwrite,
    Readonly,
}

impl From<CacheConfigMode> for CacheMode {
    fn from(value: CacheConfigMode) -> Self {
        match value {
            CacheConfigMode::Off => Self::Off,
            CacheConfigMode::Readwrite => Self::Readwrite,
            CacheConfigMode::Readonly => Self::Readonly,
        }
    }
}

impl CacheMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Readwrite => "readwrite",
            Self::Readonly => "readonly",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CacheProfile {
    pub mode: String,
    pub path: String,
    pub size_bytes: u64,
    pub max_bytes: u64,
    pub entries: usize,
    pub hits: usize,
    pub misses: usize,
    pub writes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheProfileRecord {
    pub schema_version: String,
    pub cache_profile_id: String,
    pub mode: String,
    pub path: String,
    pub entries: u64,
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
    pub total_accesses: u64,
    pub profile: CacheProfile,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheAccessRecord {
    pub schema_version: String,
    pub cache_access_id: String,
    pub sequence_number: u64,
    pub mode: String,
    pub namespace: String,
    pub boundary: String,
    pub cache_key: String,
    pub action: String,
    pub status: String,
    #[serde(default)]
    pub request_hash: Option<String>,
    #[serde(default)]
    pub response_hash: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    pub namespace: String,
    pub cache_key: String,
    pub request: Value,
    pub response: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub hit_count: u64,
}

pub struct RequestCache {
    mode: CacheMode,
    path: PathBuf,
    conn: Option<Connection>,
    schema: CacheTableSchema,
    max_bytes: u64,
    hits: usize,
    misses: usize,
    writes: usize,
    access_log: Vec<CacheAccessRecord>,
}

#[derive(Clone, Copy, Debug, Default)]
struct CacheTableSchema {
    has_metadata_json: bool,
    has_updated_at: bool,
}

impl RequestCache {
    pub fn open(path: impl AsRef<Path>, mode: CacheMode) -> Result<Self> {
        Self::open_with_max_bytes(path, mode, DEFAULT_REQUEST_CACHE_MAX_BYTES)
    }

    pub fn open_with_max_bytes(
        path: impl AsRef<Path>,
        mode: CacheMode,
        max_bytes: u64,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if mode == CacheMode::Off {
            return Ok(Self {
                mode,
                path,
                conn: None,
                schema: CacheTableSchema::default(),
                max_bytes,
                hits: 0,
                misses: 0,
                writes: 0,
                access_log: Vec::new(),
            });
        }

        if mode == CacheMode::Readwrite {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
            }
            raise_if_cache_full(&path, max_bytes, 0)?;
        }

        let conn = if path.exists() || mode == CacheMode::Readwrite {
            let conn =
                Connection::open(&path).map_err(|source| cache_corrupt(&path, "open", source))?;
            if mode == CacheMode::Readwrite {
                initialize_request_cache_schema(&path, &conn)?;
                raise_if_cache_full(&path, max_bytes, 0)?;
            }
            Some(conn)
        } else {
            None
        };
        let schema = if let Some(conn) = &conn {
            request_cache_schema(&path, conn)?
        } else {
            CacheTableSchema::default()
        };

        Ok(Self {
            mode,
            path,
            conn,
            schema,
            max_bytes,
            hits: 0,
            misses: 0,
            writes: 0,
            access_log: Vec::new(),
        })
    }

    pub fn mode(&self) -> CacheMode {
        self.mode
    }

    pub fn access_log(&self) -> &[CacheAccessRecord] {
        &self.access_log
    }

    pub fn cache_key(namespace: &str, request: &Value) -> String {
        let payload = serde_json::json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "namespace": namespace,
            "request": normalize_for_cache(request),
        });
        stable_json_hash(&payload)
    }

    pub fn cache_key_with_profile(namespace: &str, request: &Value, profile: &str) -> String {
        let payload = serde_json::json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "namespace": namespace,
            "profile": normalized_profile(profile),
            "request": normalize_for_cache_profile(request, profile),
        });
        stable_json_hash(&payload)
    }

    pub fn get_or_miss(&mut self, namespace: &str, cache_key: &str) -> Result<Option<Value>> {
        Ok(self
            .get_entry_or_miss(namespace, cache_key)?
            .map(|entry| entry.response))
    }

    pub fn get_entry_or_miss(
        &mut self,
        namespace: &str,
        cache_key: &str,
    ) -> Result<Option<CacheEntry>> {
        if self.mode == CacheMode::Off {
            return Ok(None);
        }
        let Some(conn) = &self.conn else {
            self.misses += 1;
            self.record_access(CacheAccessDraft {
                namespace,
                cache_key,
                action: "miss",
                status: "missing_cache_store",
                request: None,
                response: None,
                metadata: Map::new(),
            });
            if self.mode == CacheMode::Readonly {
                return Err(OptimizerError::CacheMiss {
                    namespace: namespace.to_string(),
                    cache_key: cache_key.to_string(),
                });
            }
            return Ok(None);
        };

        let sql = self.cache_entry_select_sql("WHERE namespace = ?1 AND cache_key = ?2", "LIMIT 1");
        let entry = conn
            .query_row(&sql, params![namespace, cache_key], cache_entry_from_row)
            .optional()
            .map_err(|source| self.cache_corrupt("get", source))?;
        if let Some(entry) = entry {
            if self.mode == CacheMode::Readwrite {
                conn.execute(
                    "UPDATE request_response_cache SET last_accessed_at = ?1, hit_count = hit_count + 1 WHERE namespace = ?2 AND cache_key = ?3",
                    params![now_seconds(), namespace, cache_key],
                )
                .map_err(|source| self.cache_corrupt("touch_hit", source))?;
            }
            self.hits += 1;
            self.record_access(CacheAccessDraft {
                namespace,
                cache_key,
                action: "hit",
                status: "completed",
                request: Some(&entry.request),
                response: Some(&entry.response),
                metadata: entry.metadata.clone(),
            });
            return Ok(Some(CacheEntry {
                hit_count: entry.hit_count + 1,
                ..entry
            }));
        }

        self.misses += 1;
        self.record_access(CacheAccessDraft {
            namespace,
            cache_key,
            action: "miss",
            status: "completed",
            request: None,
            response: None,
            metadata: Map::new(),
        });
        if self.mode == CacheMode::Readonly {
            return Err(OptimizerError::CacheMiss {
                namespace: namespace.to_string(),
                cache_key: cache_key.to_string(),
            });
        }
        Ok(None)
    }

    pub fn find_equivalent(
        &mut self,
        namespace: &str,
        request: &Value,
        profile: &str,
    ) -> Result<Option<CacheEntry>> {
        if self.mode == CacheMode::Off {
            return Ok(None);
        }
        let Some(conn) = &self.conn else {
            self.misses += 1;
            let cache_key = Self::cache_key_with_profile(namespace, request, profile);
            self.record_access(CacheAccessDraft {
                namespace,
                cache_key: &cache_key,
                action: "miss",
                status: "missing_cache_store",
                request: Some(request),
                response: None,
                metadata: cache_access_metadata(profile, Map::new()),
            });
            if self.mode == CacheMode::Readonly {
                return Err(OptimizerError::CacheMiss {
                    namespace: namespace.to_string(),
                    cache_key,
                });
            }
            return Ok(None);
        };
        let order_by = if self.schema.has_updated_at {
            "ORDER BY updated_at DESC"
        } else {
            "ORDER BY created_at DESC"
        };
        let sql = self.cache_entry_select_sql("WHERE namespace = ?1", order_by);
        let normalized_request = normalize_for_cache_profile(request, profile);
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|source| self.cache_corrupt("find_equivalent.prepare", source))?;
        let mut rows = stmt
            .query(params![namespace])
            .map_err(|source| self.cache_corrupt("find_equivalent.query", source))?;
        let mut matched = None;
        while let Some(row) = rows
            .next()
            .map_err(|source| self.cache_corrupt("find_equivalent.next", source))?
        {
            let entry =
                cache_entry_from_row(row).map_err(|source| self.cache_corrupt("decode", source))?;
            if normalize_for_cache_profile(&entry.request, profile) == normalized_request {
                matched = Some(entry);
                break;
            }
        }
        drop(rows);
        drop(stmt);
        if let Some(entry) = matched {
            if self.mode == CacheMode::Readwrite {
                conn.execute(
                    "UPDATE request_response_cache SET last_accessed_at = ?1, hit_count = hit_count + 1 WHERE namespace = ?2 AND cache_key = ?3",
                    params![now_seconds(), namespace, &entry.cache_key],
                )
                .map_err(|source| self.cache_corrupt("touch_equivalent_hit", source))?;
            }
            self.hits += 1;
            self.record_access(CacheAccessDraft {
                namespace,
                cache_key: &entry.cache_key,
                action: "hit",
                status: "equivalent",
                request: Some(&entry.request),
                response: Some(&entry.response),
                metadata: cache_access_metadata(profile, entry.metadata.clone()),
            });
            return Ok(Some(CacheEntry {
                hit_count: entry.hit_count + 1,
                ..entry
            }));
        }
        self.misses += 1;
        let cache_key = Self::cache_key_with_profile(namespace, request, profile);
        self.record_access(CacheAccessDraft {
            namespace,
            cache_key: &cache_key,
            action: "miss",
            status: "equivalent_not_found",
            request: Some(request),
            response: None,
            metadata: cache_access_metadata(profile, Map::new()),
        });
        if self.mode == CacheMode::Readonly {
            return Err(OptimizerError::CacheMiss {
                namespace: namespace.to_string(),
                cache_key,
            });
        }
        Ok(None)
    }

    pub fn put(
        &mut self,
        namespace: &str,
        cache_key: &str,
        request: &Value,
        response: &Value,
    ) -> Result<()> {
        self.put_with_metadata(
            namespace,
            cache_key,
            request,
            response,
            "generic",
            Map::new(),
        )
    }

    pub fn put_with_metadata(
        &mut self,
        namespace: &str,
        cache_key: &str,
        request: &Value,
        response: &Value,
        profile: &str,
        metadata: Map<String, Value>,
    ) -> Result<()> {
        if self.mode != CacheMode::Readwrite {
            return Ok(());
        }
        let Some(conn) = &self.conn else {
            return Ok(());
        };
        let now = now_seconds();
        let metadata = cache_access_metadata(profile, metadata);
        let request_json = stable_json(&normalize_for_cache_profile(request, profile));
        let response_json = stable_json(response);
        let metadata_json = stable_json(&Value::Object(metadata.clone()));
        raise_if_cache_full(
            &self.path,
            self.max_bytes,
            (request_json.len() + response_json.len() + metadata_json.len()) as u64,
        )?;
        conn.execute(
            r#"
            INSERT INTO request_response_cache(
                namespace, cache_key, request_json, response_json,
                metadata_json, created_at, updated_at, last_accessed_at, hit_count
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, NULL, 0)
            ON CONFLICT(namespace, cache_key) DO UPDATE SET
                request_json = excluded.request_json,
                response_json = excluded.response_json,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            "#,
            params![
                namespace,
                cache_key,
                request_json,
                response_json,
                metadata_json,
                now
            ],
        )
        .map_err(|source| self.cache_corrupt("put", source))?;
        raise_if_cache_full(&self.path, self.max_bytes, 0)?;
        self.writes += 1;
        self.record_access(CacheAccessDraft {
            namespace,
            cache_key,
            action: "write",
            status: "completed",
            request: Some(request),
            response: Some(response),
            metadata,
        });
        Ok(())
    }

    pub fn profile(&self) -> Result<CacheProfile> {
        let entries = if let Some(conn) = &self.conn {
            conn.query_row("SELECT COUNT(*) FROM request_response_cache", [], |row| {
                row.get::<_, i64>(0)
            })
            .map_err(|source| self.cache_corrupt("profile", source))? as usize
        } else {
            0
        };
        Ok(CacheProfile {
            mode: match self.mode {
                CacheMode::Off => "off",
                CacheMode::Readwrite => "readwrite",
                CacheMode::Readonly => "readonly",
            }
            .to_string(),
            path: self.path.display().to_string(),
            size_bytes: sqlite_path_size(&self.path)?,
            max_bytes: self.max_bytes,
            entries,
            hits: self.hits,
            misses: self.misses,
            writes: self.writes,
        })
    }

    fn record_access(&mut self, input: CacheAccessDraft<'_>) {
        let sequence_number = self.access_log.len() as u64 + 1;
        self.access_log
            .push(CacheAccessRecord::new(CacheAccessRecordInput {
                sequence_number,
                mode: self.mode,
                namespace: input.namespace,
                cache_key: input.cache_key,
                action: input.action,
                status: input.status,
                request: input.request,
                response: input.response,
                metadata: input.metadata,
            }));
    }

    fn cache_entry_select_sql(&self, where_clause: &str, suffix: &str) -> String {
        let metadata_expr = if self.schema.has_metadata_json {
            "metadata_json"
        } else {
            "'{}'"
        };
        format!(
            r#"
            SELECT namespace, cache_key, request_json, response_json,
                   {metadata_expr} AS metadata_json, hit_count
            FROM request_response_cache
            {where_clause}
            {suffix}
            "#
        )
    }

    fn cache_corrupt(&self, operation: &str, source: rusqlite::Error) -> OptimizerError {
        cache_corrupt(&self.path, operation, source)
    }
}

fn initialize_request_cache_schema(path: &Path, conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS request_response_cache (
            namespace TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            request_json TEXT NOT NULL,
            response_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL DEFAULT 0,
            last_accessed_at REAL,
            hit_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(namespace, cache_key)
        );
        CREATE INDEX IF NOT EXISTS idx_request_response_cache_namespace
        ON request_response_cache(namespace, updated_at);
        "#,
    )
    .map_err(|source| cache_corrupt(path, "initialize_schema", source))?;
    ensure_cache_column(path, conn, "metadata_json", "TEXT NOT NULL DEFAULT '{}'")?;
    ensure_cache_column(path, conn, "updated_at", "REAL NOT NULL DEFAULT 0")?;
    conn.execute(
        "UPDATE request_response_cache SET updated_at = created_at WHERE updated_at = 0",
        [],
    )
    .map_err(|source| cache_corrupt(path, "backfill_updated_at", source))?;
    Ok(())
}

fn ensure_cache_column(
    path: &Path,
    conn: &Connection,
    column: &str,
    column_type: &str,
) -> Result<()> {
    let schema = request_cache_schema(path, conn)?;
    let exists = match column {
        "metadata_json" => schema.has_metadata_json,
        "updated_at" => schema.has_updated_at,
        _ => false,
    };
    if exists {
        return Ok(());
    }
    conn.execute(
        &format!("ALTER TABLE request_response_cache ADD COLUMN {column} {column_type}"),
        [],
    )
    .map_err(|source| cache_corrupt(path, "alter_schema", source))?;
    Ok(())
}

fn request_cache_schema(path: &Path, conn: &Connection) -> Result<CacheTableSchema> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(request_response_cache)")
        .map_err(|source| cache_corrupt(path, "schema.prepare", source))?;
    let mut rows = stmt
        .query([])
        .map_err(|source| cache_corrupt(path, "schema.query", source))?;
    let mut schema = CacheTableSchema::default();
    while let Some(row) = rows
        .next()
        .map_err(|source| cache_corrupt(path, "schema.next", source))?
    {
        let name: String = row
            .get(1)
            .map_err(|source| cache_corrupt(path, "schema.decode", source))?;
        match name.as_str() {
            "metadata_json" => schema.has_metadata_json = true,
            "updated_at" => schema.has_updated_at = true,
            _ => {}
        }
    }
    Ok(schema)
}

fn cache_entry_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CacheEntry> {
    let request_json: String = row.get(2)?;
    let response_json: String = row.get(3)?;
    let metadata_json: String = row.get(4)?;
    let request = serde_json::from_str(&request_json).unwrap_or(Value::Null);
    let response = serde_json::from_str(&response_json).unwrap_or(Value::Null);
    let metadata_value = serde_json::from_str(&metadata_json).unwrap_or(Value::Object(Map::new()));
    let metadata = metadata_value.as_object().cloned().unwrap_or_else(Map::new);
    Ok(CacheEntry {
        namespace: row.get(0)?,
        cache_key: row.get(1)?,
        request,
        response,
        metadata,
        hit_count: row.get::<_, i64>(5)?.max(0) as u64,
    })
}

fn cache_corrupt(path: &Path, operation: &str, source: rusqlite::Error) -> OptimizerError {
    OptimizerError::CacheCorrupt {
        path: path.to_path_buf(),
        operation: operation.to_string(),
        message: source.to_string(),
    }
}

fn raise_if_cache_full(path: &Path, max_bytes: u64, extra_bytes: u64) -> Result<()> {
    if max_bytes == 0 {
        return Ok(());
    }
    let size_bytes = sqlite_path_size(path)?.saturating_add(extra_bytes);
    if size_bytes >= max_bytes {
        return Err(OptimizerError::CacheFull {
            path: path.to_path_buf(),
            size_bytes,
            max_bytes,
        });
    }
    Ok(())
}

fn sqlite_path_size(path: &Path) -> Result<u64> {
    let mut total = file_size(path)?;
    let wal_path = sidecar_sqlite_path(path, "wal");
    total = total.saturating_add(file_size(&wal_path)?);
    let shm_path = sidecar_sqlite_path(path, "shm");
    total = total.saturating_add(file_size(&shm_path)?);
    Ok(total)
}

fn sidecar_sqlite_path(path: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}-{suffix}", path.display()))
}

fn file_size(path: &Path) -> Result<u64> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.len()),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(source) => Err(OptimizerError::io(path, source)),
    }
}

fn cache_access_metadata(profile: &str, mut metadata: Map<String, Value>) -> Map<String, Value> {
    metadata
        .entry("cache_profile".to_string())
        .or_insert_with(|| Value::String(normalized_profile(profile).to_string()));
    metadata
}

fn normalized_profile(profile: &str) -> &str {
    let profile = profile.trim();
    if profile.is_empty() {
        "generic"
    } else {
        profile
    }
}

impl CacheProfileRecord {
    pub fn from_profile(profile: CacheProfile) -> Self {
        let total_accesses = profile.hits + profile.misses + profile.writes;
        Self {
            schema_version: "cache_profile_record.v1".to_string(),
            cache_profile_id: stable_id("cacheprofile", &[&profile.path, &profile.mode]),
            mode: profile.mode.clone(),
            path: profile.path.clone(),
            entries: profile.entries as u64,
            hits: profile.hits as u64,
            misses: profile.misses as u64,
            writes: profile.writes as u64,
            total_accesses: total_accesses as u64,
            profile,
        }
    }
}

impl CacheAccessRecord {
    fn new(input: CacheAccessRecordInput<'_>) -> Self {
        let request_hash = input.request.map(stable_value_hash);
        let response_hash = input.response.map(stable_value_hash);
        let boundary = cache_boundary(input.namespace);
        Self {
            schema_version: "cache_access_record.v1".to_string(),
            cache_access_id: stable_id(
                "cacheaccess",
                &[
                    &input.sequence_number.to_string(),
                    input.namespace,
                    input.cache_key,
                    input.action,
                ],
            ),
            sequence_number: input.sequence_number,
            mode: input.mode.as_str().to_string(),
            namespace: input.namespace.to_string(),
            boundary,
            cache_key: input.cache_key.to_string(),
            action: input.action.to_string(),
            status: input.status.to_string(),
            request_hash,
            response_hash,
            metadata: input.metadata,
        }
    }
}

struct CacheAccessRecordInput<'a> {
    sequence_number: u64,
    mode: CacheMode,
    namespace: &'a str,
    cache_key: &'a str,
    action: &'a str,
    status: &'a str,
    request: Option<&'a Value>,
    response: Option<&'a Value>,
    metadata: Map<String, Value>,
}

struct CacheAccessDraft<'a> {
    namespace: &'a str,
    cache_key: &'a str,
    action: &'a str,
    status: &'a str,
    request: Option<&'a Value>,
    response: Option<&'a Value>,
    metadata: Map<String, Value>,
}

pub fn stable_json(value: &Value) -> String {
    serde_json::to_string(&sort_json(value)).unwrap_or_else(|_| "null".to_string())
}

pub fn normalize_for_cache(value: &Value) -> Value {
    normalize_for_cache_profile(value, "generic")
}

pub fn normalize_for_cache_profile(value: &Value, profile: &str) -> Value {
    normalize_for_cache_value(value, normalized_profile(profile))
}

fn normalize_for_cache_value(value: &Value, profile: &str) -> Value {
    match value {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, item) in map {
                let key_lower = key.to_ascii_lowercase();
                if is_secret_key(&key_lower) {
                    out.insert(key.clone(), Value::String("<redacted>".to_string()));
                    continue;
                }
                if is_volatile_key(key, profile) {
                    continue;
                }
                out.insert(key.clone(), normalize_for_cache_value(item, profile));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .filter(|item| !is_cache_only_event(item, profile))
                .map(|item| normalize_for_cache_value(item, profile))
                .collect(),
        ),
        Value::String(text) => Value::String(normalize_text(text)),
        _ => value.clone(),
    }
}

fn sort_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            let mut sorted = Map::new();
            for key in keys {
                if let Some(item) = map.get(key) {
                    sorted.insert(key.clone(), sort_json(item));
                }
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_json).collect()),
        _ => value.clone(),
    }
}

fn is_secret_key(key_lower: &str) -> bool {
    key_lower.contains("authorization")
        || key_lower.contains("api_key")
        || key_lower.contains("apikey")
        || key_lower.contains("secret")
        || key_lower.contains("password")
        || matches!(
            key_lower,
            "token" | "access_token" | "refresh_token" | "id_token"
        )
        || key_lower.ends_with("_api_token")
}

fn is_volatile_key(key: &str, profile: &str) -> bool {
    let key = key.to_ascii_lowercase();
    if matches!(
        key.as_str(),
        "ts" | "timestamp"
            | "created_at"
            | "updated_at"
            | "last_accessed_at"
            | "run_id"
            | "rollout_id"
            | "rollout_ids"
            | "cache_key"
            | "cache_hit"
            | "request_hash"
            | "artifact_id"
            | "artifact_ids"
            | "artifact_ref"
            | "artifact_refs"
            | "trace_correlation_id"
            | "correlation_id"
            | "request_correlation_id"
            | "session_id"
            | "thread_id"
            | "call_id"
            | "request_id"
            | "job_id"
            | "container_rollout_id"
            | "frame_id"
            | "trace_id"
            | "trace_ref"
            | "trace_refs"
            | "trial_id"
            | "source_rollout_id"
            | "source_call_id"
            | "sensor_frame_id"
            | "input_ref"
            | "output_ref"
            | "produced_by"
            | "storage_uri"
            | "started_at"
            | "completed_at"
            | "duration_s"
            | "latency_s"
            | "elapsed_s"
            | "wall_s"
            | "timeout_s"
            | "timeout_seconds"
            | "request_timeout_s"
            | "request_timeout_seconds"
            | "upstream_timeout_s"
            | "upstream_timeout_seconds"
            | "cost_usd"
            | "proxy_trace_path"
            | "materializer_id"
            | "workspace"
            | "workspace_dir"
            | "workspace_root"
            | "artifact_root"
            | "output_dir"
            | "run_dir"
            | "manifest_path"
            | "event_feed_path"
            | "normalized_event_feed_path"
            | "cache_profile_path"
            | "best_candidate_path"
            | "candidate_registry_path"
            | "frontier_path"
            | "run_registry_path"
    ) {
        return true;
    }
    if matches!(
        profile,
        "rollout_request" | "subagent_invocation" | "gepa_workspace"
    ) {
        if matches!(
            key.as_str(),
            "artifacts" | "links" | "link_id" | "workspace_pack_manifest"
        ) {
            return true;
        }
        if profile == "rollout_request" && matches!(key.as_str(), "context" | "submission_mode") {
            return true;
        }
        if key == "total_cost_usd" || key.ends_with("_cost_usd") {
            return true;
        }
        if key.ends_with("_path") || key.ends_with("_root") || key.ends_with("_dir") {
            return true;
        }
        if matches!(
            key.as_str(),
            "usage" | "usage_json" | "token_usage" | "events" | "trace_events"
        ) {
            return true;
        }
    }
    false
}

fn is_cache_only_event(value: &Value, profile: &str) -> bool {
    if !matches!(profile, "subagent_invocation" | "gepa_workspace") {
        return false;
    }
    let Value::Object(map) = value else {
        return false;
    };
    let kind = map
        .get("kind")
        .or_else(|| map.get("type"))
        .or_else(|| map.get("event_type"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    matches!(
        kind.as_str(),
        "evaluation_cache_hit"
            | "platform_cache_hit"
            | "platform_cache_miss"
            | "platform_cache_write"
    )
}

fn normalize_text(text: &str) -> String {
    text.to_string()
}

fn cache_boundary(namespace: &str) -> String {
    namespace
        .rsplit_once(':')
        .map(|(_, boundary)| boundary)
        .unwrap_or(namespace)
        .to_string()
}

pub fn stable_value_hash(value: &Value) -> String {
    sha256_text(&stable_json(&normalize_for_cache(value)))
}

pub fn stable_json_hash(value: &Value) -> String {
    sha256_text(&stable_json(value))
}

fn stable_id(prefix: &str, parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(prefix.as_bytes());
    for part in parts {
        digest.update(b"\0");
        digest.update(part.as_bytes());
    }
    let hex = format!("{:x}", digest.finalize());
    format!("{prefix}_{}", &hex[..16])
}

fn sha256_text(text: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(text.as_bytes());
    format!("{:x}", digest.finalize())
}

fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}
