use uuid::Uuid;

use crate::error::Error;

/// Parsed callback data from an inline keyboard button press.
#[derive(Debug, Clone, PartialEq)]
pub enum CallbackAction {
    /// Tool approval decision: `a:{uuid}:{decision}`
    Approval {
        interaction_id: Uuid,
        decision: String,
    },
    /// Question answer: `q:{uuid}:{question_idx}:{option_idx}`
    QuestionAnswer {
        interaction_id: Uuid,
        question_idx: usize,
        option_idx: usize,
    },
}

/// Build inline keyboard markup data for a tool approval prompt.
///
/// Returns `Vec<(label, callback_data)>` pairs for each button.
pub fn approval_buttons(interaction_id: Uuid) -> Vec<(String, String)> {
    let id = interaction_id.to_string();
    vec![
        ("Allow".into(), format!("a:{id}:allow")),
        ("Deny".into(), format!("a:{id}:deny")),
        ("Always Allow".into(), format!("a:{id}:always_allow")),
    ]
}

/// Build inline keyboard markup data for a question prompt.
///
/// Returns `Vec<(label, callback_data)>` pairs for each option across all questions.
pub fn question_buttons(
    interaction_id: Uuid,
    questions: &[(String, Vec<String>)],
) -> Vec<(String, String)> {
    let id = interaction_id.to_string();
    let mut buttons = Vec::new();
    for (q_idx, (_question, options)) in questions.iter().enumerate() {
        for (o_idx, label) in options.iter().enumerate() {
            buttons.push((label.clone(), format!("q:{id}:{q_idx}:{o_idx}")));
        }
    }
    buttons
}

/// Parse a callback_data string into a `CallbackAction`.
pub fn parse_callback_data(data: &str) -> Result<CallbackAction, Error> {
    let parts: Vec<&str> = data.splitn(4, ':').collect();
    match parts.first() {
        Some(&"a") => {
            if parts.len() != 3 {
                return Err(Error::Telegram(format!(
                    "invalid approval callback: expected 3 parts, got {}",
                    parts.len()
                )));
            }
            let interaction_id = Uuid::parse_str(parts[1])
                .map_err(|e| Error::Telegram(format!("invalid UUID in callback: {e}")))?;
            Ok(CallbackAction::Approval {
                interaction_id,
                decision: parts[2].to_string(),
            })
        }
        Some(&"q") => {
            if parts.len() != 4 {
                return Err(Error::Telegram(format!(
                    "invalid question callback: expected 4 parts, got {}",
                    parts.len()
                )));
            }
            let interaction_id = Uuid::parse_str(parts[1])
                .map_err(|e| Error::Telegram(format!("invalid UUID in callback: {e}")))?;
            let question_idx: usize = parts[2]
                .parse()
                .map_err(|e| Error::Telegram(format!("invalid question index: {e}")))?;
            let option_idx: usize = parts[3]
                .parse()
                .map_err(|e| Error::Telegram(format!("invalid option index: {e}")))?;
            Ok(CallbackAction::QuestionAnswer {
                interaction_id,
                question_idx,
                option_idx,
            })
        }
        _ => Err(Error::Telegram(format!("unknown callback prefix: {data}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approval_buttons_layout() {
        let id = Uuid::new_v4();
        let buttons = approval_buttons(id);
        assert_eq!(buttons.len(), 3);
        assert_eq!(buttons[0].0, "Allow");
        assert_eq!(buttons[1].0, "Deny");
        assert_eq!(buttons[2].0, "Always Allow");
    }

    #[test]
    fn approval_callback_data_format() {
        let id = Uuid::new_v4();
        let buttons = approval_buttons(id);
        assert!(buttons[0].1.starts_with("a:"));
        assert!(buttons[0].1.ends_with(":allow"));
        assert!(buttons[1].1.ends_with(":deny"));
        assert!(buttons[2].1.ends_with(":always_allow"));
    }

    #[test]
    fn approval_callback_roundtrip() {
        let id = Uuid::new_v4();
        let buttons = approval_buttons(id);
        for (_, data) in &buttons {
            let action = parse_callback_data(data).unwrap();
            match action {
                CallbackAction::Approval {
                    interaction_id,
                    decision,
                } => {
                    assert_eq!(interaction_id, id);
                    assert!(!decision.is_empty());
                }
                _ => panic!("expected Approval, got: {action:?}"),
            }
        }
    }

    #[test]
    fn question_buttons_layout() {
        let id = Uuid::new_v4();
        let questions = vec![
            ("Pick a color".into(), vec!["Red".into(), "Blue".into()]),
            ("Pick a size".into(), vec!["Small".into(), "Large".into()]),
        ];
        let buttons = question_buttons(id, &questions);
        assert_eq!(buttons.len(), 4);
        assert_eq!(buttons[0].0, "Red");
        assert_eq!(buttons[1].0, "Blue");
        assert_eq!(buttons[2].0, "Small");
        assert_eq!(buttons[3].0, "Large");
    }

    #[test]
    fn question_callback_roundtrip() {
        let id = Uuid::new_v4();
        let questions = vec![
            ("Q1".into(), vec!["A".into(), "B".into()]),
            ("Q2".into(), vec!["C".into()]),
        ];
        let buttons = question_buttons(id, &questions);

        // First question, second option
        let action = parse_callback_data(&buttons[1].1).unwrap();
        assert_eq!(
            action,
            CallbackAction::QuestionAnswer {
                interaction_id: id,
                question_idx: 0,
                option_idx: 1,
            }
        );

        // Second question, first option
        let action = parse_callback_data(&buttons[2].1).unwrap();
        assert_eq!(
            action,
            CallbackAction::QuestionAnswer {
                interaction_id: id,
                question_idx: 1,
                option_idx: 0,
            }
        );
    }

    #[test]
    fn parse_malformed_prefix() {
        let err = parse_callback_data("x:something").unwrap_err();
        assert!(err.to_string().contains("unknown callback prefix"));
    }

    #[test]
    fn parse_malformed_approval_missing_parts() {
        let err = parse_callback_data("a:only-uuid").unwrap_err();
        assert!(err.to_string().contains("invalid approval callback"));
    }

    #[test]
    fn parse_malformed_approval_invalid_uuid() {
        let err = parse_callback_data("a:not-a-uuid:allow").unwrap_err();
        assert!(err.to_string().contains("invalid UUID"));
    }

    #[test]
    fn parse_malformed_question_missing_parts() {
        let id = Uuid::new_v4();
        let err = parse_callback_data(&format!("q:{id}:0")).unwrap_err();
        assert!(err.to_string().contains("invalid question callback"));
    }

    #[test]
    fn parse_malformed_question_invalid_index() {
        let id = Uuid::new_v4();
        let err = parse_callback_data(&format!("q:{id}:abc:0")).unwrap_err();
        assert!(err.to_string().contains("invalid question index"));
    }

    #[test]
    fn parse_empty_string() {
        let err = parse_callback_data("").unwrap_err();
        assert!(err.to_string().contains("unknown callback prefix"));
    }

    #[test]
    fn question_buttons_empty_questions() {
        let id = Uuid::new_v4();
        let buttons = question_buttons(id, &[]);
        assert!(buttons.is_empty());
    }

    #[test]
    fn question_buttons_single_option() {
        let id = Uuid::new_v4();
        let questions = vec![("Q".into(), vec!["Only".into()])];
        let buttons = question_buttons(id, &questions);
        assert_eq!(buttons.len(), 1);
        assert_eq!(buttons[0].0, "Only");
    }
}
