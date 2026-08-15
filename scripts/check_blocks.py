#!/usr/bin/env python3
"""Static block-balance checker for Julia sources.

This is NOT a parser.  It strips comments, strings and char literals, then checks
that block-opening keywords balance against `end`, tracking bracket depth so that
`x[end]` and `x[1:end-1]` are not mistaken for block terminators.  It catches the
single most common generation defect (a missing or extra `end`) without a Julia
runtime available.
"""
import re, sys, os

OPENERS = {"function", "if", "for", "while", "let", "struct", "begin", "quote",
           "do", "module", "try", "macro"}
# `if` after `else`/`elseif` handled naturally; `end` closes one level.

def strip_code(src):
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '#':
            m0 = i
            if src.startswith("#=", i):
                depth = 1; i += 2
                while i < n and depth:
                    if src.startswith("#=", i): depth += 1; i += 2
                    elif src.startswith("=#", i): depth -= 1; i += 2
                    else: i += 1
                out.append('\n' * src.count('\n', m0, i))
                continue
            while i < n and src[i] != '\n': i += 1
            continue
        if src.startswith('"""', i):
            m0 = i
            i += 3
            while i < n and not src.startswith('"""', i):
                if src[i] == '\\': i += 1
                i += 1
            i += 3
            out.append(' STR ' + '\n' * src.count('\n', m0, i))
            continue
        if c == '"':
            m0 = i
            i += 1
            while i < n and src[i] != '"':
                if src[i] == '\\': i += 1
                i += 1
            i += 1
            out.append(' STR ' + '\n' * src.count('\n', m0, i))
            continue
        if c == "'":
            m = re.match(r"'(\\.|[^'\\])'", src[i:])
            if m:
                i += m.end(); out.append(' CHR '); continue
        out.append(c)
        i += 1
    return ''.join(out)

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_!]*|.", re.S)

def check(path):
    src = open(path, encoding='utf-8').read()
    code = strip_code(src)
    # precompute line starts for accurate reporting
    line_starts = [0]
    for i, ch in enumerate(code):
        if ch == '\n':
            line_starts.append(i + 1)

    def lineno(pos):
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    depth = 0
    bracket = 0
    paren = 0
    stack = []
    problems = []
    prev = ''
    for m in TOKEN.finditer(code):
        t = m.group(0)
        if t == '[':
            bracket += 1
        elif t == ']':
            bracket -= 1
        elif t == '(':
            paren += 1
        elif t == ')':
            paren -= 1
        elif prev not in ('.', ':') and (bracket == 0 and paren == 0):
            # Inside () or [] a `for`/`if` is a comprehension or generator and a
            # bare `end` is the index keyword: neither participates in blocks.
            # A keyword after `:` is a Symbol literal (`:function`, `:for`).
            if t in OPENERS or (t == 'type' and prev in ('abstract', 'primitive')):
                depth += 1
                stack.append((t, lineno(m.start())))
            elif t == 'end':
                depth -= 1
                if stack:
                    stack.pop()
                else:
                    problems.append(f"{path}:{lineno(m.start())}: `end` with no open block")
        if t.strip():
            prev = t
    if depth != 0:
        problems.append(f"{path}: block depth {depth:+d} at EOF; open: " +
                        ", ".join(f"{k}@{l}" for k, l in stack[-6:]))
    if paren != 0:
        problems.append(f"{path}: parenthesis imbalance {paren:+d}")
    if bracket != 0:
        problems.append(f"{path}: bracket imbalance {bracket:+d}")
    nonascii = [(i + 1, l) for i, l in enumerate(src.split('\n'))
                if any(ord(ch) > 127 for ch in l)]
    for ln, l in nonascii[:3]:
        problems.append(f"{path}:{ln}: non-ASCII character in source")
    return problems

if __name__ == '__main__':
    files = []
    for root in sys.argv[1:]:
        if os.path.isdir(root):
            for d, _, fs in os.walk(root):
                files += [os.path.join(d, f) for f in fs if f.endswith('.jl')]
        else:
            files.append(root)
    allp = []
    for f in sorted(files):
        p = check(f)
        allp += p
        print(f"{'FAIL' if p else 'ok  '}  {f}")
    for p in allp:
        print("  !", p)
    sys.exit(1 if allp else 0)
