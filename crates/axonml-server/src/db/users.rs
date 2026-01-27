//! User database operations for AxonML
//!
//! Uses Aegis-DB Document Store for user management.

use super::{Database, DbError, DocumentQuery};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Collection name for users
const COLLECTION: &str = "axonml_users";

/// User role enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    Admin,
    User,
    Viewer,
}

impl Default for UserRole {
    fn default() -> Self {
        UserRole::User
    }
}

/// User data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub email: String,
    pub name: String,
    pub password_hash: String,
    #[serde(default)]
    pub role: UserRole,
    #[serde(default)]
    pub mfa_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub totp_secret: Option<String>,
    #[serde(default)]
    pub webauthn_credentials: Vec<serde_json::Value>,
    #[serde(default)]
    pub recovery_codes: Vec<String>,
    #[serde(default = "default_email_pending")]
    pub email_pending: bool,
    #[serde(default)]
    pub email_verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification_token: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

fn default_email_pending() -> bool {
    true
}

/// New user creation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewUser {
    pub email: String,
    pub name: String,
    pub password_hash: String,
    #[serde(default)]
    pub role: UserRole,
}

/// User update data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpdateUser {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<UserRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mfa_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub totp_secret: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webauthn_credentials: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recovery_codes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email_pending: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email_verified: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification_token: Option<String>,
}

/// User repository for database operations
pub struct UserRepository<'a> {
    db: &'a Database,
}

impl<'a> UserRepository<'a> {
    /// Create a new user repository
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Create a new user
    pub async fn create(&self, new_user: NewUser) -> Result<User, DbError> {
        // Check if email already exists using document store query
        let existing = self.find_by_email(&new_user.email).await?;
        if existing.is_some() {
            return Err(DbError::AlreadyExists(format!(
                "User with email {} already exists",
                new_user.email
            )));
        }

        let now = Utc::now();
        let verification_token = Uuid::new_v4().to_string();
        let user = User {
            id: Uuid::new_v4().to_string(),
            email: new_user.email,
            name: new_user.name,
            password_hash: new_user.password_hash,
            role: new_user.role,
            mfa_enabled: false,
            totp_secret: None,
            webauthn_credentials: vec![],
            recovery_codes: vec![],
            email_pending: true,
            email_verified: false,
            verification_token: Some(verification_token),
            created_at: now,
            updated_at: now,
        };

        let user_json = serde_json::to_value(&user)?;

        // Insert using document store
        self.db
            .doc_insert(COLLECTION, Some(&user.id), user_json)
            .await?;

        Ok(user)
    }

    /// Find user by ID
    pub async fn find_by_id(&self, id: &str) -> Result<Option<User>, DbError> {
        let doc = self.db.doc_get(COLLECTION, id).await?;

        match doc {
            Some(data) => {
                let user: User = serde_json::from_value(data)?;
                Ok(Some(user))
            }
            None => Ok(None),
        }
    }

    /// Find user by email
    pub async fn find_by_email(&self, email: &str) -> Result<Option<User>, DbError> {
        // Use document store filter with $eq operator
        let filter = serde_json::json!({
            "email": { "$eq": email }
        });

        let doc = self.db.doc_find_one(COLLECTION, filter).await?;

        match doc {
            Some(data) => {
                let user: User = serde_json::from_value(data)?;
                Ok(Some(user))
            }
            None => Ok(None),
        }
    }

    /// Find user by name (username)
    pub async fn find_by_name(&self, name: &str) -> Result<Option<User>, DbError> {
        // Use document store filter with $eq operator
        let filter = serde_json::json!({
            "name": { "$eq": name }
        });

        let doc = self.db.doc_find_one(COLLECTION, filter).await?;

        match doc {
            Some(data) => {
                let user: User = serde_json::from_value(data)?;
                Ok(Some(user))
            }
            None => Ok(None),
        }
    }

    /// Update user
    pub async fn update(&self, id: &str, update: UpdateUser) -> Result<User, DbError> {
        let mut user = self
            .find_by_id(id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("User {} not found", id)))?;

        // Apply updates
        if let Some(name) = update.name {
            user.name = name;
        }
        if let Some(email) = update.email {
            user.email = email;
        }
        if let Some(password_hash) = update.password_hash {
            user.password_hash = password_hash;
        }
        if let Some(role) = update.role {
            user.role = role;
        }
        if let Some(mfa_enabled) = update.mfa_enabled {
            user.mfa_enabled = mfa_enabled;
        }
        if let Some(totp_secret) = update.totp_secret {
            user.totp_secret = Some(totp_secret);
        }
        if let Some(webauthn_credentials) = update.webauthn_credentials {
            user.webauthn_credentials = webauthn_credentials;
        }
        if let Some(recovery_codes) = update.recovery_codes {
            user.recovery_codes = recovery_codes;
        }
        if let Some(email_pending) = update.email_pending {
            user.email_pending = email_pending;
        }
        if let Some(email_verified) = update.email_verified {
            user.email_verified = email_verified;
        }
        if let Some(verification_token) = update.verification_token {
            user.verification_token = Some(verification_token);
        }

        user.updated_at = Utc::now();

        let user_json = serde_json::to_value(&user)?;

        // Update using document store
        self.db.doc_update(COLLECTION, id, user_json).await?;

        Ok(user)
    }

    /// Delete user
    pub async fn delete(&self, id: &str) -> Result<(), DbError> {
        // Check if user exists first
        let _ = self
            .find_by_id(id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("User {} not found", id)))?;

        self.db.doc_delete(COLLECTION, id).await?;

        Ok(())
    }

    /// List all users
    pub async fn list(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<User>, DbError> {
        let query = DocumentQuery {
            filter: None,
            sort: Some(serde_json::json!({ "field": "created_at", "ascending": false })),
            limit,
            skip: offset,
        };

        let docs = self.db.doc_query(COLLECTION, query).await?;

        let mut users = Vec::new();
        for doc in docs {
            let user: User = serde_json::from_value(doc)?;
            users.push(user);
        }

        Ok(users)
    }

    /// Count total users
    pub async fn count(&self) -> Result<u64, DbError> {
        // Query all users and count them
        let query = DocumentQuery {
            filter: None,
            sort: None,
            limit: None,
            skip: None,
        };

        let docs = self.db.doc_query(COLLECTION, query).await?;
        Ok(docs.len() as u64)
    }

    /// Enable TOTP for user
    pub async fn enable_totp(&self, id: &str, secret: &str) -> Result<User, DbError> {
        self.update(
            id,
            UpdateUser {
                mfa_enabled: Some(true),
                totp_secret: Some(secret.to_string()),
                ..Default::default()
            },
        )
        .await
    }

    /// Disable MFA for user
    pub async fn disable_mfa(&self, id: &str) -> Result<User, DbError> {
        let mut user = self
            .find_by_id(id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("User {} not found", id)))?;

        user.mfa_enabled = false;
        user.totp_secret = None;
        user.webauthn_credentials = vec![];
        user.recovery_codes = vec![];
        user.updated_at = Utc::now();

        let user_json = serde_json::to_value(&user)?;

        self.db.doc_update(COLLECTION, id, user_json).await?;

        Ok(user)
    }

    /// Set recovery codes for user
    pub async fn set_recovery_codes(&self, id: &str, codes: Vec<String>) -> Result<User, DbError> {
        self.update(
            id,
            UpdateUser {
                recovery_codes: Some(codes),
                ..Default::default()
            },
        )
        .await
    }

    /// Use a recovery code
    pub async fn use_recovery_code(&self, id: &str, code_hash: &str) -> Result<bool, DbError> {
        let mut user = self
            .find_by_id(id)
            .await?
            .ok_or_else(|| DbError::NotFound(format!("User {} not found", id)))?;

        // Find and remove the code
        let original_len = user.recovery_codes.len();
        user.recovery_codes.retain(|c| c != code_hash);

        if user.recovery_codes.len() == original_len {
            return Ok(false); // Code not found
        }

        user.updated_at = Utc::now();
        let user_json = serde_json::to_value(&user)?;

        self.db.doc_update(COLLECTION, id, user_json).await?;

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_serialization() {
        let user = User {
            id: "test-123".to_string(),
            email: "test@example.com".to_string(),
            name: "Test User".to_string(),
            password_hash: "hash".to_string(),
            role: UserRole::User,
            mfa_enabled: false,
            totp_secret: None,
            webauthn_credentials: vec![],
            recovery_codes: vec![],
            email_pending: true,
            email_verified: false,
            verification_token: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let json = serde_json::to_string(&user).unwrap();
        assert!(json.contains("test@example.com"));
        assert!(json.contains("\"role\":\"user\""));
    }

    #[test]
    fn test_user_role_default() {
        let role = UserRole::default();
        assert_eq!(role, UserRole::User);
    }
}
