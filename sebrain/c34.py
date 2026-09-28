                    )
                    try:
                        payload = json.loads(candidate)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict) and (
                        payload.get("record_id") or payload.get("id")
                    ):
                        candidates.append(candidate)

        # Code-bearing fields can contain many nested string literals. Try
        # each plausible JSON field boundary rather than assuming the first
        # comma-delimited quote is structural; accept only a candidate that
        # parses completely and retains a record identity.
        code_field_names = (
            "source_code", "correct_code", "incorrect_code", "corrected_code",
            "corrected_implementation", "incorrect_implementation", "test_code",
            "test_input", "source", "code", "example", "invalid_example",
        )
        for key in code_field_names:
            marker = '"' + key + '":"'
            key_positions = []
            key_start = line.find(marker)
            while key_start >= 0:
                key_positions.append(key_start)
                key_start = line.find(marker, key_start + len(marker))
            for key_start in key_positions:
                if error_pos is not None and error_pos < key_start:
                    continue
                value_start = key_start + len(marker)
                boundary = line.find('","', value_start)
                if error_pos is not None and error_pos > value_start:
                    boundary = line.find('","', error_pos)
                attempts = 0
                while boundary >= 0 and attempts < 24:
                    value = line[value_start:boundary]
                    if '"' in value:
                        candidate = (
                            line[:value_start]
                            + value.replace('"', '\\\"')
                            + line[boundary:]
                        )
                        try:
                            payload = json.loads(candidate)
                        except json.JSONDecodeError:
                            payload = None
                        if isinstance(payload, dict) and (
                            payload.get("record_id") or payload.get("id")
                        ) and key in payload:
                            candidates.append(candidate)
                    boundary = line.find('","', boundary + 3)
                    attempts += 1

        # Conservative recovery for a malformed final string field whose
        # value contains raw double quotes (common in exported code examples).
        field_marker = '":"'
        field_start = line.rfind(field_marker)
        # Embedded code/string concatenations may occur in any field.
        # Preserve their literal semantics by escaping only the quote characters
        # within the containing field, then require complete JSON validation.