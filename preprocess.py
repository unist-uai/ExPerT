import argparse
import pandas as pd
import numpy as np
import os

def load_data(file_path):
    print(f"[INFO] Loading data from {file_path}")
    df = pd.read_csv(file_path, na_filter = False)

    # Force timestamp columns to integer type
    time_cols = ["key_down_time", "key_up_time"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def save_data(df, output_path):
    print(f"[INFO] Saving data to {output_path}")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

def prepreprocess_data(df):

    print("[INFO] preprocessing data")
    for col in ["key_down_time", "key_up_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")


    # ======================================================
    # 1) Base sorting (only within each user + query)
    # ======================================================
    df = df.sort_values(
        ['user_id', 'query', 'key_down_time']
    ).reset_index(drop=True)

    # Keep the existing exception handling
    df.loc[df['session_id'].str.startswith('p1_1751'), 'user_id'] = 'p2'

    # ======================================================
    # 2) Assign word_group
    # ======================================================
    df = assign_word_group_by_query(df)

    # ======================================================
    # 3) Build events_df
    # ======================================================
    events_df = df[
        ['key_name', 'key_down_time', 'key_up_time',
        'query', 'user_id', 'expertise', 'word_group', 'task']
    ].copy()

    # ======================================================
    # 4) Create pairs (groupby + shift ONLY)
    # ======================================================
    g = events_df.groupby(['user_id', 'query'])

    events_df['key2'] = g['key_name'].shift(-1)
    events_df['key2_down_time'] = g['key_down_time'].shift(-1)
    events_df['key2_up_time'] = g['key_up_time'].shift(-1)
    events_df['key2_word_group'] = g['word_group'].shift(-1)

    # ======================================================
    # 5) Pair dataframe
    # ======================================================
    pairs_df = events_df.rename(columns={
        'key_name': 'key1',
        'key_down_time': 'key1_down_time',
        'key_up_time': 'key1_up_time',
        'word_group': 'key1_word_group'
    })

    # Remove the last pair explicitly
    pairs_df = pairs_df.dropna(
        subset=['key2_down_time', 'key2_up_time']
    ).reset_index(drop=True)

    # ======================================================
    # 6) word_group assignment rule
    # ======================================================
    pairs_df['word_group'] = pairs_df['key1_word_group']
    pairs_df.loc[
        pairs_df['key2_word_group'] != pairs_df['key1_word_group'],
        'word_group'
    ] = pairs_df['key2_word_group']

    time_cols = [
        "key1_down_time", "key1_up_time",
        "key2_down_time", "key2_up_time"
    ]

    for col in time_cols:
        pairs_df[col] = pd.to_numeric(pairs_df[col], errors="coerce").astype("Int64")

    # ======================================================
    # 7) Compute features
    # ======================================================
    pairs_df['Duration_key1'] = pairs_df['key1_up_time'] - pairs_df['key1_down_time']
    pairs_df['Duration_key2'] = pairs_df['key2_up_time'] - pairs_df['key2_down_time']
    pairs_df['DD_time'] = pairs_df['key2_down_time'] - pairs_df['key1_down_time']
    pairs_df['UD_time'] = pairs_df['key2_down_time'] - pairs_df['key1_up_time']
    pairs_df['UU_time'] = pairs_df['key2_up_time'] - pairs_df['key1_up_time']
    pairs_df['DU_time'] = pairs_df['key2_up_time'] - pairs_df['key1_down_time']

    pairs_df.to_csv("./data/preprocessed_raw.csv", index=False, encoding="utf-8-sig")

    return pairs_df


def preprocess_word_level(df):
    print("[INFO] Preprocessing word-level data")
    # 0) Adjust the last key to the last word in the query
    def fix_last_word(df):
        df = df.copy()
        for (u, q), g in df.groupby(["user_id", "query"]):
            if len(g) <= 1:
                continue
            last_idx = g.index[-1]
            prev_idx = g.index[-2]
            df.loc[last_idx, "word_group"] = df.loc[prev_idx, "word_group"]
        return df

    df = fix_last_word(df)

    # 1) DataFrame that preserves the original order
    order_df = (
        df[['query', 'user_id', 'expertise', 'word_group', 'task']]
        .drop_duplicates()
        .reset_index()
        .rename(columns={'index': 'orig_order'})
    )



    # 1) Aggregate duration and timing features
    word_user_features_df = df.groupby(['query', 'user_id', 'expertise', 'word_group', 'task']).agg({
        'Duration_key1': ['mean', 'std'],
        'Duration_key2': ['mean', 'std'],
        'DD_time': ['mean', 'std'],
        'UD_time': ['mean', 'std'],
        'UU_time': ['mean', 'std'],
        'DU_time': ['mean', 'std'],
    }).reset_index()

    word_user_features_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in word_user_features_df.columns
    ]

    # 2) Backspace count
    backspace_counts = (
        df[df['key1'] == 'Backspace']
        .groupby(['query', 'user_id', 'expertise', 'word_group', 'task'])
        .size()
        .reset_index(name='Backspace_count')
    )

    # 3) Typing speed
    key_counts = (
        df.groupby(['query', 'user_id', 'expertise', 'word_group', 'task'])
        .size()
        .reset_index(name='Total_keys')
    )

    time_range = (
        df.groupby(['query', 'user_id', 'expertise', 'word_group', 'task'])
        .agg(
            start_time=('key1_down_time', 'min'),
            end_time=('key1_down_time', 'max')
        )
        .reset_index()
    )

    time_range['Duration_minutes'] = (time_range['end_time'] - time_range['start_time']) / (1000 * 60)

    typing_speed = time_range.merge(
        key_counts,
        on=['query', 'user_id', 'expertise', 'word_group', 'task']
    )
    typing_speed['Typing_speed_kpm'] = typing_speed['Total_keys'] / typing_speed['Duration_minutes']

    typing_speed = typing_speed[['query', 'user_id', 'expertise', 'word_group', 'task', 'Typing_speed_kpm']]

    # 4) Merge the results
    result_df = (
        word_user_features_df
        .merge(backspace_counts, on=['query', 'user_id', 'expertise', 'word_group', 'task'], how='left')
        .merge(typing_speed, on=['query', 'user_id', 'expertise', 'word_group', 'task'], how='left')
    )

    # 5) Fill NaN values in Backspace_count
    result_df['Backspace_count'] = result_df['Backspace_count'].fillna(0).astype(int)

    # 6) Restore the original order
    result_df = (
        order_df
        .merge(
            result_df,
            on=['query', 'user_id', 'expertise', 'word_group', 'task'],
            how='left'
        )
        .sort_values('orig_order')
        .reset_index(drop=True)
    )
    
    result_df['typing_features'] = result_df.apply(format_typing_features, axis=1)
    return result_df


def preprocess_query_level(df):
    # Existing aggregation
    query_user_features_df = df.groupby(['query', 'user_id', 'expertise', 'task']).agg({
        'Duration_key1': ['mean', 'std'],
        'Duration_key2': ['mean', 'std'],
        'DD_time': ['mean', 'std'],
        'UD_time': ['mean', 'std'],
        'UU_time': ['mean', 'std'],
        'DU_time': ['mean', 'std'],
    }).reset_index()

    query_user_features_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in query_user_features_df.columns
    ]

    # Backspace count
    backspace_counts = (
        df[df['key1'] == 'Backspace']
        .groupby(['query', 'user_id', 'task'])
        .size()
        .reset_index(name='Backspace_count')
    )

    # Typing speed
    key_counts = (
        df.groupby(['query', 'user_id', 'task'])
        .size()
        .reset_index(name='Total_keys')
    )

    time_range = (
        df.groupby(['query', 'user_id', 'task'])
        .agg(
            start_time=('key1_down_time', 'min'),
            end_time=('key1_down_time', 'max')
        )
        .reset_index()
    )

    time_range['Duration_minutes'] = (time_range['end_time'] - time_range['start_time']) / (1000 * 60)

    typing_speed = time_range.merge(key_counts, on=['query', 'user_id', 'task'])
    typing_speed['Typing_speed_kpm'] = typing_speed['Total_keys'] / typing_speed['Duration_minutes']

    typing_speed = typing_speed[['query', 'user_id', 'task', 'Typing_speed_kpm']]

    # Merge aggregated results
    result_df = (
        query_user_features_df
        .merge(backspace_counts, on=['query', 'user_id', 'task'], how='left')
        .merge(typing_speed, on=['query', 'user_id', 'task'], how='left')
    )



    # Fill NaN values in Backspace_count with 0
    result_df['Backspace_count'] = result_df['Backspace_count'].fillna(0).astype(int)

    # Build the formatted output column
    result_df['typing_features'] = result_df.apply(format_typing_features, axis=1)
    return result_df


def format_typing_features(row):
    return (
        f"• Duration: mean={row['Duration_key1_mean']:.2f} / std={row['Duration_key1_std']:.2f}\n"
        f"• DD-time: mean={row['DD_time_mean']:.2f} / std={row['DD_time_std']:.2f}\n"
        f"• UD-time: mean={row['UD_time_mean']:.2f} / std={row['UD_time_std']:.2f}\n"
        f"• UU-time: mean={row['UU_time_mean']:.2f} / std={row['UU_time_std']:.2f}\n"
        f"• DU-time: mean={row['DU_time_mean']:.2f} / std={row['DU_time_std']:.2f}\n"
        f"• Total number of Backspace keys: {row['Backspace_count']}\n"
        f"• Typing speed (keystrokes per minute): {row['Typing_speed_kpm']:.2f}\n"
    )

def decompose_hangul_syllable(char):
    CHO = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
    JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
    JONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
    
    code = ord(char)
    if code < 0xAC00 or code > 0xD7A3:
        return [char]
    
    syllable_index = code - 0xAC00
    jong_index = syllable_index % 28
    jung_index = ((syllable_index - jong_index)//28)%21
    cho_index = ((syllable_index - jong_index)//28)//21

    cho = CHO[cho_index]
    jung = JUNG[jung_index]
    jong = JONG[jong_index]

    result = [cho]
    # Handle compound vowels
    if jung in ["ㅘ"]:
        result.extend(["ㅗ","ㅏ"])
    elif jung == "ㅙ":
        result.extend(["ㅗ","ㅐ"])
    elif jung=="ㅚ":
        result.extend(["ㅗ","ㅣ"])
    elif jung=="ㅝ":
        result.extend(["ㅜ","ㅓ"])
    elif jung=="ㅞ":
        result.extend(["ㅜ","ㅔ"])
    elif jung=="ㅟ":
        result.extend(["ㅜ","ㅣ"])
    elif jung=="ㅢ":
        result.extend(["ㅡ","ㅣ"])
    else:
        result.append(jung)

    if jong != "":
        result.append(jong)

    return result



DOUBLE_KOR_MAP = {
    "ㄲ": "ㄱ",
    "ㄸ": "ㄷ",
    "ㅃ": "ㅂ",
    "ㅆ": "ㅅ",
    "ㅉ": "ㅈ",
    "ㅒ": "ㅐ",
    "ㅖ": "ㅔ",
}

def key_to_analysis_jamo(key: str) -> str | None:
    """
    Key normalization for word_group comparison:
    - lowercase letters
    - remove shift
    - use the base two-set Korean jamo
    """
    if not isinstance(key, str):
        return None

    k = key.strip()

    # Exclude special keys
    if len(k) != 1:
        return None

    # English letters -> lowercase
    if k.isascii() and k.isalpha():
        return k.lower()

    # Double Korean jamo -> base jamo
    if k in DOUBLE_KOR_MAP:
        return DOUBLE_KOR_MAP[k]

    return k

def word_to_jamos(word):
    """Convert a word into a jamo sequence (split Hangul, keep others as-is)."""

    jamos = []
    for ch in word:
        parts = decompose_hangul_syllable(ch)
        for p in parts:
            # English letters -> lowercase
            if isinstance(p, str) and p.isascii() and p.isalpha():
                jamos.append(p.lower())

            # Double Korean jamo -> base jamo
            elif p in DOUBLE_KOR_MAP:
                jamos.append(DOUBLE_KOR_MAP[p])

            # Others (numbers, symbols, etc.)
            else:
                jamos.append(p)

    return jamos


def assign_word_group_by_query(df):
    WORD_BOUNDARY_KEYS = {
        " ", "Space",
        "Enter", "ENTER", "enter",
        "Return", "RETURN",
        "KeypadEnter", "NumpadEnter",
    }

    df = df.copy()
    df.sort_values(by=["user_id", "session_id", "key_down_time"], inplace=True)
    df["word_group"] = None

    for (user, session), group_df in df.groupby(["user_id", "session_id"]):
        query = group_df["query"].iloc[0]
        words = query.split()
        if not words:
            continue

        # Prepare jamo sequences for all words
        words_jamo = [word_to_jamos(w) for w in words]

        word_idx = 0
        current_word = words[word_idx]
        target_jamo = words_jamo[word_idx]
        match_index = -1  # subsequence pointer

        # Use the first jamo of the first word as the start condition
        first_word_first_jamo = target_jamo[0]
        first_word_started = False  # Flag that determines the dummy section

        for idx in group_df.index:
            key = str(df.loc[idx, "key_name"])

            # --------------------------------
            # 0) Before the first word starts -> always dummy
            # --------------------------------
            if not first_word_started:
                # Start once the first jamo of the first word is typed
                if key == first_word_first_jamo:
                    first_word_started = True
                    match_index = 0
                    df.loc[idx, "word_group"] = current_word
                else:
                    df.loc[idx, "word_group"] = "dummy"
                continue

            # --------------------------------
            # 1) Backspace
            # --------------------------------
            else: 
                if key == "Backspace":
                    df.loc[idx, "word_group"] = current_word
                    continue

                # --------------------------------
                # 2) Space/Enter -> word boundary
                # --------------------------------
                if key in WORD_BOUNDARY_KEYS:
                    if match_index + 1 == len(target_jamo):
                        # Word completed -> move to the next word
                        word_idx += 1
                        if word_idx < len(words):
                            current_word = words[word_idx]
                            target_jamo = words_jamo[word_idx]
                            match_index = -1
                        else:
                            # If it moves past the final word, keep the final word fixed
                            word_idx = len(words) - 1
                            current_word = words[-1]
                            target_jamo = words_jamo[-1]
                            match_index = len(target_jamo) - 1

                    df.loc[idx, "word_group"] = current_word
                    continue

                # --------------------------------
                # 3) Regular character -> jamo subsequence match
                # --------------------------------
                if len(key) == 1:
                    if match_index + 1 < len(target_jamo):
                        next_need = target_jamo[match_index + 1]
                        if key == next_need:
                            match_index += 1

                    df.loc[idx, "word_group"] = current_word
                    continue

                df.loc[idx, "word_group"] = current_word

        # Force the last key to belong to the last word
        last_idx = group_df.index[-1]
        df.loc[last_idx, "word_group"] = df.loc[last_idx -1, "word_group"]

    return df

def assign_word_group_by_query(df):
    # import pandas as pd
    from difflib import SequenceMatcher

    CURSOR_KEYS = {
        "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
        "Home", "End"
    }
    SPACE_KEYS = {" ", "Space"}
    IGNORE_KEYS = {"shift"}  # modifier only
    EDIT_KEYS = {"Backspace", "Delete"}

    df = df.copy()
    df.sort_values(by=["user_id", "session_id", "key_down_time"], inplace=True)
    df["word_group"] = None

    # --------------------------
    # helper: build spans on QUERY TOKENS
    # --------------------------
    def build_query_spans(words):
        spans = []
        pos = 0
        for w in words:
            t = word_to_jamos(w)  # <- already normalized (double-jamo simplified, english lower, etc.)
            start = pos
            end = start + len(t)
            spans.append((start, end, w))
            pos = end + 1  # assume single space between words
        return spans

    def find_word_by_qpos(qpos, spans):
        for s, e, w in spans:
            if s <= qpos < e:
                return w
        if qpos < spans[0][0]:
            return spans[0][2]
        return spans[-1][2]

    # --------------------------
    # helper: align buffer_tokens to query_tokens
    # build map: buffer_index -> query_index (approx)
    # --------------------------
    def build_buf_to_query_map(buffer_tokens, query_tokens):
        """
        Returns list map_q where map_q[i] = approx query index corresponding to buffer token i.
        If buffer is empty, returns empty list.
        """
        n = len(buffer_tokens)
        if n == 0:
            return []

        # SequenceMatcher works on sequences of hashable items (tokens ok)
        sm = SequenceMatcher(a=buffer_tokens, b=query_tokens, autojunk=False)
        opcodes = sm.get_opcodes()

        # default: map to 0
        map_q = [0] * n

        # We'll walk through ops and assign a piecewise-linear mapping.
        # For equal blocks: exact alignment.
        # For replace/insert/delete: approximate with nearest b-index.
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                # buffer[i1:i2] matches query[j1:j2] one-to-one
                for k in range(i2 - i1):
                    map_q[i1 + k] = j1 + k
            elif tag == "delete":
                # buffer chunk deleted w.r.t query -> map to j1 (nearest)
                for i in range(i1, i2):
                    map_q[i] = max(0, min(j1, len(query_tokens) - 1)) if query_tokens else 0
            elif tag == "insert":
                # nothing in buffer; skip
                continue
            elif tag == "replace":
                # map buffer chunk to query chunk roughly (stretch/compress)
                b_len = i2 - i1
                q_len = j2 - j1
                if b_len <= 0:
                    continue
                if q_len <= 0:
                    # no target in query, map to j1-1
                    tgt = max(0, min(j1 - 1, len(query_tokens) - 1)) if query_tokens else 0
                    for i in range(i1, i2):
                        map_q[i] = tgt
                else:
                    # linear mapping
                    for t in range(b_len):
                        # t in [0, b_len-1] -> mapped to [j1, j2-1]
                        jj = j1 + int(round(t * (q_len - 1) / max(1, b_len - 1)))
                        map_q[i1 + t] = max(0, min(jj, len(query_tokens) - 1))

        return map_q

    # ==============================
    # session loop
    # ==============================
    for (user, session), group_df in df.groupby(["user_id", "session_id"]):
        query = group_df["query"].iloc[0]
        words = query.split()
        if not words:
            continue

        # Query tokens & spans (token index coordinate system)
        query_tokens = []
        for wi, w in enumerate(words):
            query_tokens.extend(word_to_jamos(w))
            if wi < len(words) - 1:
                query_tokens.append(" ")  # represent word boundary as a token too

        spans = build_query_spans(words)

        # editor state (buffer tokens are also in same token space)
        buffer = []
        caret = 0
        prev_word_group = None

                # ===== [ADD] State for word tracking =====
        intent_word = None            # The word currently being maintained
        typed_tokens = []             # The "word tokens" currently being typed


        first_query_token = query_tokens[0] if len(query_tokens) > 0 else None
        started_typing = False

        # alignment cache (updated when needed)
        buf2q = []
        alignment_dirty = True  # mark dirty whenever buffer/caret edit occurs

        def current_word_group_from_alignment(caret_pos):
            """
            caret_pos: buffer index (token position where next insert happens)
            Convert caret_pos -> query_pos using buf2q mapping.
            """
            nonlocal buf2q, alignment_dirty
            if alignment_dirty:
                buf2q = build_buf_to_query_map(buffer, query_tokens)
                alignment_dirty = False

            # If buffer empty, fallback to first word
            if not buffer:
                return spans[0][2]

            # caret points *between* tokens. Choose a representative token index:
            # - for insertion, use caret-1 (previous token) if possible, else 0
            rep = min(max(caret_pos - 1, 0), len(buffer) - 1)

            # map to query token index
            qpos = buf2q[rep] if rep < len(buf2q) else min(len(query_tokens) - 1, 0)
            # print(qpos)
            return find_word_by_qpos(qpos, spans)

        # iterate events
        for idx in group_df.index:
            raw = df.loc[idx, "key_name"]
            key = " " if pd.isna(raw) else str(raw)
            # print(key)

            # --------------------------------
            # 0) Before the first actual typing -> dummy
            # --------------------------------
            if not started_typing:
                # Compare with the first query token only for character input
                if len(key) == 1 and first_query_token is not None:
                    key_tokens = word_to_jamos(key)  # Convert into the same token space as query_tokens
                    # Start if key_tokens is not empty and its first token matches the query's first token
                    if key_tokens and key_tokens[0] == first_query_token:
                        started_typing = True
                        # Exit dummy mode -> continue into the logic below (do not continue here)
                    else:
                        df.loc[idx, "word_group"] = "<dummy>"
                        continue
                else:
                    df.loc[idx, "word_group"] = "<dummy>"
                    continue


            # ignore modifier
            if key in IGNORE_KEYS:
                df.loc[idx, "word_group"] = prev_word_group or current_word_group_from_alignment(caret)
                continue

            # cursor movement (doesn't change buffer, but affects caret)
            if key in CURSOR_KEYS:
                if key == "ArrowLeft":
                    caret = max(0, caret - 1)
                elif key == "ArrowRight":
                    caret = min(len(buffer), caret + 1)
                elif key == "Home":
                    caret = 0
                elif key == "End":
                    caret = len(buffer)

                # df.loc[idx, "word_group"] = current_word_group_from_alignment(caret)
                # prev_word_group = df.loc[idx, "word_group"]

                df.loc[idx, "word_group"] = "<navigation>"
                prev_word_group = current_word_group_from_alignment(caret)
                continue

            # backspace/delete (buffer changes => alignment dirty)
            if key == "Backspace":
                if caret > 0:
                    del buffer[caret - 1]
                    caret -= 1
                    alignment_dirty = True
                    
                    # ===== [ADD] Remove the word token as well =====
                    if typed_tokens:
                        typed_tokens.pop()
                # df.loc[idx, "word_group"] = current_word_group_from_alignment(caret)
                df.loc[idx, "word_group"] = prev_word_group
                prev_word_group = df.loc[idx, "word_group"]
                continue

            if key == "Delete":
                if caret < len(buffer):
                    del buffer[caret]
                    alignment_dirty = True
                df.loc[idx, "word_group"] = current_word_group_from_alignment(caret)
                prev_word_group = df.loc[idx, "word_group"]
                continue

            # space: insert token + inherit the previous word_group
            if key in SPACE_KEYS:
                buffer.insert(caret, " ")
                caret += 1
                alignment_dirty = True

                # df.loc[idx, "word_group"] = prev_word_group or current_word_group_from_alignment(caret)
                df.loc[idx, "word_group"] = prev_word_group
                prev_word_group = df.loc[idx, "word_group"]
                continue

            # normal char
            if len(key) == 1:
                buffer.insert(caret, key)
                caret += 1
                alignment_dirty = True

                df.loc[idx, "word_group"] = current_word_group_from_alignment(caret)
                prev_word_group = df.loc[idx, "word_group"]
                continue

            # fallback
            df.loc[idx, "word_group"] = prev_word_group or current_word_group_from_alignment(caret)
            prev_word_group = df.loc[idx, "word_group"]

    return df

def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save processed CSV")
    parser.add_argument("--preprocess_mode", type=str, choices=["word", "query"], required=True,
                        help="Preprocessing mode")

    args = parser.parse_args()

    df = load_data(args.input_path)
    df = prepreprocess_data(df)  # Preprocessing step is required

    if args.preprocess_mode == "word":
        df = preprocess_word_level(df)
    elif args.preprocess_mode == "query":
        df = preprocess_query_level(df)

    save_data(df, args.output_path)
    print("[INFO] Preprocessing completed successfully.")



if __name__ == "__main__":
    main()
