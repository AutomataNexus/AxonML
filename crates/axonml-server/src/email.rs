//! Email service using Resend

use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmailError {
    #[error("Failed to send email: {0}")]
    SendError(String),
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),
    #[error("Email service not configured - RESEND_API_KEY not set")]
    NotConfigured,
}

#[derive(Debug, Serialize)]
struct ResendEmailRequest {
    from: String,
    to: Vec<String>,
    subject: String,
    html: String,
}

#[derive(Debug, Deserialize)]
struct ResendEmailResponse {
    id: String,
}

pub struct EmailService {
    api_key: Option<String>,
    client: Client,
    from_email: String,
}

impl EmailService {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            client: Client::new(),
            from_email: "AxonML <noreply@automatanexus.com>".to_string(),
        }
    }

    /// Check if email service is properly configured
    pub fn is_configured(&self) -> bool {
        self.api_key.is_some()
    }

    /// Send verification email to user
    pub async fn send_verification_email(
        &self,
        to_email: &str,
        user_name: &str,
        verification_token: &str,
        base_url: &str,
    ) -> Result<(), EmailError> {
        let verify_url = format!("{}/api/auth/verify-email?token={}", base_url, verification_token);

        let html = format!(
            r#"
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Verify Your Email - AxonML</title>
            </head>
            <body style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #111827; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #faf9f6;">
                <div style="background-color: #ffffff; border-radius: 12px; padding: 40px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #14b8a6; font-size: 32px; margin: 0;">AxonML</h1>
                    </div>

                    <h2 style="color: #111827; font-size: 24px; margin-bottom: 16px;">Welcome to AxonML, {}!</h2>

                    <p style="color: #6b7280; font-size: 16px; margin-bottom: 24px;">
                        Thank you for signing up! Please verify your email address to continue. Once verified,
                        an administrator will review and approve your account.
                    </p>

                    <div style="text-align: center; margin: 32px 0;">
                        <a href="{}" style="display: inline-block; background-color: #14b8a6; color: #ffffff; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-weight: 600; font-size: 16px;">
                            Verify Email Address
                        </a>
                    </div>

                    <p style="color: #9ca3af; font-size: 14px; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
                        If the button doesn't work, copy and paste this link into your browser:<br>
                        <a href="{}" style="color: #14b8a6; word-break: break-all;">{}</a>
                    </p>

                    <p style="color: #9ca3af; font-size: 14px; margin-top: 16px;">
                        If you didn't create an account with AxonML, you can safely ignore this email.
                    </p>

                    <div style="text-align: center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
                        <p style="color: #9ca3af; font-size: 12px; margin: 4px 0;">
                            Secured by AutomataNexus
                        </p>
                        <p style="color: #9ca3af; font-size: 12px; margin: 4px 0;">
                            © 2026 AxonML. All rights reserved.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            "#,
            user_name, verify_url, verify_url, verify_url
        );

        self.send_email(to_email, "Verify Your Email - AxonML", &html).await
    }

    /// Send notification to admin about new user signup
    pub async fn send_admin_signup_notification(
        &self,
        user_email: &str,
        user_name: &str,
    ) -> Result<(), EmailError> {
        let html = format!(
            r#"
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>New User Signup - AxonML</title>
            </head>
            <body style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #111827; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #faf9f6;">
                <div style="background-color: #ffffff; border-radius: 12px; padding: 40px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #14b8a6; font-size: 32px; margin: 0;">AxonML</h1>
                    </div>

                    <h2 style="color: #111827; font-size: 24px; margin-bottom: 16px;">New User Signup</h2>

                    <p style="color: #6b7280; font-size: 16px; margin-bottom: 24px;">
                        A new user has registered for AxonML and is awaiting email verification:
                    </p>

                    <div style="background-color: #f0fdfa; border-left: 4px solid #14b8a6; padding: 16px; margin: 24px 0; border-radius: 4px;">
                        <p style="margin: 8px 0;"><strong>Name:</strong> {}</p>
                        <p style="margin: 8px 0;"><strong>Email:</strong> {}</p>
                    </div>

                    <p style="color: #6b7280; font-size: 14px; margin-top: 24px;">
                        Once the user verifies their email, you'll receive another notification to approve their access.
                    </p>

                    <div style="text-align: center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
                        <p style="color: #9ca3af; font-size: 12px; margin: 4px 0;">
                            Secured by AutomataNexus
                        </p>
                    </div>
                </div>
            </body>
            </html>
            "#,
            user_name, user_email
        );

        self.send_email("devops@automatanexus.com", "New User Signup - AxonML", &html).await
    }

    /// Send approval request to admin after email verification
    pub async fn send_admin_approval_request(
        &self,
        user_id: &str,
        user_email: &str,
        user_name: &str,
        user_location: Option<&str>,
        user_ip: Option<&str>,
        approval_token: &str,
        base_url: &str,
    ) -> Result<(), EmailError> {
        let approval_url = format!("{}/api/auth/approve-user?token={}", base_url, approval_token);

        let location_info = user_location.unwrap_or("Unknown");
        let ip_info = user_ip.unwrap_or("Unknown");

        let html = format!(
            r#"
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>User Approval Required - AxonML</title>
            </head>
            <body style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #111827; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #faf9f6;">
                <div style="background-color: #ffffff; border-radius: 12px; padding: 40px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #14b8a6; font-size: 32px; margin: 0;">AxonML</h1>
                    </div>

                    <h2 style="color: #111827; font-size: 24px; margin-bottom: 16px;">User Approval Required</h2>

                    <p style="color: #6b7280; font-size: 16px; margin-bottom: 24px;">
                        A user has verified their email address and is requesting access to AxonML:
                    </p>

                    <div style="background-color: #f0fdfa; border-left: 4px solid #14b8a6; padding: 16px; margin: 24px 0; border-radius: 4px;">
                        <p style="margin: 8px 0;"><strong>Username:</strong> {}</p>
                        <p style="margin: 8px 0;"><strong>Email:</strong> {}</p>
                        <p style="margin: 8px 0;"><strong>Full Name:</strong> {}</p>
                        <p style="margin: 8px 0;"><strong>Location:</strong> {}</p>
                        <p style="margin: 8px 0;"><strong>IP Address:</strong> {}</p>
                        <p style="margin: 8px 0;"><strong>User ID:</strong> {}</p>
                    </div>

                    <div style="text-align: center; margin: 32px 0;">
                        <a href="{}" style="display: inline-block; background-color: #14b8a6; color: #ffffff; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-weight: 600; font-size: 16px;">
                            Permit {} Access
                        </a>
                    </div>

                    <p style="color: #9ca3af; font-size: 14px; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
                        If the button doesn't work, copy and paste this link into your browser:<br>
                        <a href="{}" style="color: #14b8a6; word-break: break-all;">{}</a>
                    </p>

                    <div style="text-align: center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
                        <p style="color: #9ca3af; font-size: 12px; margin: 4px 0;">
                            Secured by AutomataNexus
                        </p>
                    </div>
                </div>
            </body>
            </html>
            "#,
            user_name, user_email, user_name, location_info, ip_info, user_id,
            approval_url, user_name, approval_url, approval_url
        );

        self.send_email("devops@automatanexus.com", &format!("Approval Required: {} - AxonML", user_name), &html).await
    }

    /// Send welcome email after approval
    pub async fn send_welcome_email(
        &self,
        to_email: &str,
        user_name: &str,
        dashboard_url: &str,
    ) -> Result<(), EmailError> {
        let html = format!(
            r#"
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Welcome to AxonML</title>
            </head>
            <body style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #111827; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #faf9f6;">
                <div style="background-color: #ffffff; border-radius: 12px; padding: 40px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #14b8a6; font-size: 32px; margin: 0;">AxonML</h1>
                    </div>

                    <h2 style="color: #111827; font-size: 24px; margin-bottom: 16px;">Welcome, {}!</h2>

                    <p style="color: #6b7280; font-size: 16px; margin-bottom: 24px;">
                        Your account has been approved! You can now access the AxonML platform and start
                        building amazing ML models.
                    </p>

                    <div style="text-align: center; margin: 32px 0;">
                        <a href="{}" style="display: inline-block; background-color: #14b8a6; color: #ffffff; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-weight: 600; font-size: 16px;">
                            Access Dashboard
                        </a>
                    </div>

                    <div style="background-color: #f0fdfa; border-left: 4px solid #14b8a6; padding: 16px; margin: 24px 0; border-radius: 4px;">
                        <h3 style="color: #111827; font-size: 16px; margin-top: 0;">Getting Started</h3>
                        <ul style="color: #6b7280; font-size: 14px; margin: 0; padding-left: 20px;">
                            <li>Explore the training dashboard</li>
                            <li>Upload your first model</li>
                            <li>Deploy inference endpoints</li>
                            <li>Monitor metrics and performance</li>
                        </ul>
                    </div>

                    <p style="color: #6b7280; font-size: 14px; margin-top: 24px;">
                        If you have any questions, feel free to reach out to our support team.
                    </p>

                    <div style="text-align: center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
                        <p style="color: #9ca3af; font-size: 12px; margin: 4px 0;">
                            Secured by AutomataNexus
                        </p>
                        <p style="color: #9ca3af; font-size: 12px; margin: 4px 0;">
                            © 2026 AxonML. All rights reserved.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            "#,
            user_name, dashboard_url
        );

        self.send_email(to_email, "Welcome to AxonML - Your Account is Active!", &html).await
    }

    /// Internal method to send email via Resend API
    async fn send_email(
        &self,
        to: &str,
        subject: &str,
        html: &str,
    ) -> Result<(), EmailError> {
        // Check if API key is configured
        let api_key = self.api_key.as_ref().ok_or(EmailError::NotConfigured)?;

        let request = ResendEmailRequest {
            from: self.from_email.clone(),
            to: vec![to.to_string()],
            subject: subject.to_string(),
            html: html.to_string(),
        };

        let response = self.client
            .post("https://api.resend.com/emails")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(EmailError::SendError(error_text));
        }

        let result: ResendEmailResponse = response.json().await?;
        tracing::debug!(email_id = %result.id, to = to, "Email sent successfully");
        Ok(())
    }
}
