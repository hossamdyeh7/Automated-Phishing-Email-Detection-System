# phishing_detector_app.py
import os
import re
import time
import json
import base64
import threading
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import imapclient
import email
from bs4 import BeautifulSoup
import requests

# Optional ML imports (only needed if you plan to train / run the model)
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# -------------------------
# Config / Globals
# -------------------------
PHISHING_KEYWORDS = [
    'urgent', 'suspicious activity', 'account suspension', 'reset your password now',
    'confirm your account now', 'claim your', 'gift card', 'free trial', 'password reset',
    'account locked', 'unauthorized access', 'password has been compromised', 'update your',
    'invoice overdue', 'click here', 'limited time offer',
    'bank alert', 'login now', 'immediate action',
    'confirm your identity', 'potential fraud', 'win a prize', 'update payment'
]

# These will be set from the login UI
EMAIL = ""
PASSWORD = ""

# Model & tokenizer (global cache)
model = None
tokenizer = None

# Cached emails fetched from IMAP
emails_list = []

# Path for assets (use relative path; put your PNGs in ./assets/frame0/)
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "assets" / "frame0"


# -------------------------
# Utility functions
# -------------------------
def contains_phishing_keywords(text: str) -> bool:
    text = (text or "").lower()
    for keyword in PHISHING_KEYWORDS:
        # word boundary search to reduce false positives
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            return True
    return False


# -------------------------
# IMAP / Email fetching
# -------------------------
def fetch_emails():
    """
    Fetch emails from the IMAP server using IMAPClient.
    Returns: list of dicts with keys: id, subject, from, date, body, full_msg
    """
    global emails_list, EMAIL, PASSWORD
    emails_list = []

    if not EMAIL or not PASSWORD:
        messagebox.showerror("Error", "EMAIL or PASSWORD not set.")
        return []

    try:
        # IMAPClient with ssl=True
        with imapclient.IMAPClient("imap.gmail.com", ssl=True) as server:
            server.login(EMAIL, PASSWORD)
            server.select_folder("INBOX", readonly=True)

            messages = server.search(["ALL"])
            if not messages:
                return []

            fetch_data = server.fetch(messages, ["RFC822", "INTERNALDATE"])
            for msgid, data in fetch_data.items():
                raw_email = data.get(b"RFC822")
                if not raw_email:
                    continue

                msg = email.message_from_bytes(raw_email)

                # Extract a readable body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        disp = part.get("Content-Disposition", "")
                        # prefer html, then text, skip attachments
                        if "attachment" in disp:
                            continue
                        if content_type == "text/html":
                            try:
                                body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                                break
                            except Exception:
                                continue
                        elif content_type == "text/plain" and not body:
                            try:
                                body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                            except Exception:
                                continue
                else:
                    try:
                        body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                    except Exception:
                        body = str(msg.get_payload())

                # INTERNALDATE safe extraction
                date_val = data.get(b"INTERNALDATE")
                if date_val:
                    try:
                        date = date_val.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        date = str(date_val)
                else:
                    date = "Unknown"

                emails_list.append({
                    "id": int(msgid),
                    "subject": msg.get("subject") or "No Subject",
                    "from": msg.get("from") or "Unknown",
                    "date": date,
                    "body": body,
                    "full_msg": msg
                })

            # sort by date string (ISO-like) - if date string unknown, put later
            emails_list.sort(key=lambda x: x["date"] if x["date"] != "Unknown" else "1970-01-01 00:00:00", reverse=True)
            return emails_list

    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch emails: {e}")
        return []


# -------------------------
# ML: Dataset and training (optional)
# -------------------------
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = int(self.labels[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def train_model_threadsafe(callback=None, epochs=2):
    """
    Train or load model in a background thread. If saved model exists, loads it.
    callback: optional function to call after done (called on main thread ideally).
    """
    thread = threading.Thread(target=_train_model_worker, args=(callback, epochs), daemon=True)
    thread.start()
    return thread


def _train_model_worker(callback, epochs):
    """
    Worker that trains/loads the model. This executes off the UI thread.
    """
    global model, tokenizer
    try:
        save_dir = OUTPUT_PATH / "saved_model"

        # Load saved model if exists
        if save_dir.exists():
            print("Loading saved model/tokenizer...")
            tokenizer = RobertaTokenizer.from_pretrained(str(save_dir))
            model = RobertaForSequenceClassification.from_pretrained(str(save_dir))
            if callback:
                try:
                    callback(success=True, message="Loaded saved model.")
                except Exception:
                    pass
            return

        # Check dataset exists
        dataset_path = OUTPUT_PATH / "email_dataset.csv"
        if not dataset_path.exists():
            if callback:
                callback(success=False, message=f"Dataset not found: {dataset_path}")
            return

        df = pd.read_csv(str(dataset_path))
        if "label" not in df.columns:
            if callback:
                callback(success=False, message="Dataset missing 'label' column.")
            return

        df['combined_text'] = df['subject'].fillna('') + " " + df['body'].fillna('')

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['combined_text'], df['label'], test_size=0.2, random_state=42
        )

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

        MAX_LEN = 256
        BATCH_SIZE = 16

        train_dataset = EmailDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, MAX_LEN)
        test_dataset = EmailDataset(test_texts.tolist(), test_labels.tolist(), tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            model.train()
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                running_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Avg train loss: {running_loss / len(train_loader)}")

            # Evaluation
            model.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, predicted = torch.max(outputs.logits, dim=1)
                    preds.extend(predicted.cpu().tolist())
                    actuals.extend(labels.cpu().tolist())

            acc = accuracy_score(actuals, preds)
            print(f"Validation accuracy after epoch {epoch+1}: {acc}")

        # Save model and tokenizer
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))

        if callback:
            callback(success=True, message="Model trained and saved.")

    except Exception as e:
        if callback:
            callback(success=False, message=f"Training error: {e}")
        else:
            print("Training error:", e)


# -------------------------
# VirusTotal helper
# -------------------------
def vt_scan_url(url, result_callback):
    """
    Submit URL to VirusTotal and fetch results. Runs in a thread.
    result_callback(result_text) will be called with status/result updates (string).
    """
    api_key = os.getenv("VT_API_KEY")
    if not api_key:
        result_callback("VirusTotal API key not set (set VT_API_KEY environment variable).")
        return

    headers = {"x-apikey": api_key}
    submit_url_endpoint = "https://www.virustotal.com/api/v3/urls"

    try:
        resp = requests.post(submit_url_endpoint, headers=headers, data={"url": url}, timeout=15)
        if resp.status_code != 200 and resp.status_code != 201:
            result_callback(f"Failed to submit URL: {resp.status_code} {resp.text}")
            return

        # Extract id from response if available, else generate from url encoding
        # According to VT, the resource id is in data.id sometimes. Using URL encoding is also common.
        try:
            resp_json = resp.json()
            # id may be in resp_json["data"]["id"]
            vt_id = resp_json.get("data", {}).get("id")
        except Exception:
            vt_id = None

        if not vt_id:
            vt_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")

        analysis_url = f"https://www.virustotal.com/api/v3/urls/{vt_id}"

        # Wait a short time and poll
        result_callback("Waiting for analysis (this may take ~10-30s)...")
        for _ in range(10):
            time.sleep(3)  # poll every 3s
        # now try to fetch result
        r2 = requests.get(analysis_url, headers=headers, timeout=15)
        if r2.status_code != 200:
            result_callback(f"Error getting analysis: {r2.status_code} {r2.text}")
            return

        data = r2.json()
        stats = data["data"]["attributes"]["last_analysis_stats"]
        result_text = f"URL: {url}\n"
        result_text += f"Malicious: {stats.get('malicious',0)}\n"
        result_text += f"Suspicious: {stats.get('suspicious',0)}\n"
        result_text += f"Undetected: {stats.get('undetected',0)}\n"
        result_text += f"Harmless: {stats.get('harmless',0)}\n\n"
        # Detailed per-engine
        results = data["data"]["attributes"].get("last_analysis_results", {})
        for engine, details in results.items():
            result_text += f"{engine}: {details.get('category','unknown')}"
            if details.get('result'):
                result_text += f" ({details['result']})"
            result_text += "\n"

        result_callback(result_text)

    except Exception as e:
        result_callback(f"VirusTotal scanning error: {e}")


# -------------------------
# UI Functions
# -------------------------
def start_training_and_notify():
    def cb(success, message):
        # This callback runs in training thread; schedule UI update on main thread
        def ui_notify():
            if success:
                messagebox.showinfo("Training", message)
            else:
                messagebox.showerror("Training", message)
        root.after(0, ui_notify)

    train_model_threadsafe(callback=cb, epochs=2)


def check_email_threaded():
    """Run check_email in a background thread to avoid blocking UI."""
    thread = threading.Thread(target=check_email, daemon=True)
    thread.start()


def check_email():
    """
    Fetch emails, run a quick keyword check first, then model prediction if available.
    Updates the global email_tree (UI) and emails_list.
    """
    global model, tokenizer, email_tree, emails_list

    # run fetch_emails in worker thread? Here already called from a thread wrapper
    emails = fetch_emails()
    if not emails:
        # show a small msg or just return
        return

    # update UI tree - must run on main thread using root.after
    def ui_update():
        # clear tree
        for it in email_tree.get_children():
            email_tree.delete(it)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model:
            try:
                _ = model.to(device)
                model.eval()
            except Exception:
                pass

        for em in emails:
            combined_text = (em.get("subject") or "") + " " + (em.get("body") or "")
            if contains_phishing_keywords(combined_text):
                status = "Phishing (Keyword)"
                color = "red"
            else:
                if model and tokenizer:
                    try:
                        encoding = tokenizer.encode_plus(
                            combined_text,
                            add_special_tokens=True,
                            max_length=256,
                            return_token_type_ids=False,
                            padding="max_length",
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors="pt",
                        )
                        input_ids = encoding['input_ids'].to(device)
                        attention_mask = encoding['attention_mask'].to(device)
                        with torch.no_grad():
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            _, pred = torch.max(outputs.logits, dim=1)
                            pred = int(pred.cpu().item())
                        status = "Phishing" if pred == 1 else "Safe"
                        color = "red" if pred == 1 else "green"
                    except Exception:
                        status = "Error (model)"
                        color = "black"
                else:
                    status = "Unknown"
                    color = "black"

            # use msg id as iid so we can find it easily later
            iid = str(em["id"])
            preview = BeautifulSoup(em.get("body", "") or "", "html.parser").get_text()[:120]
            email_tree.insert("", "end", iid=iid, values=(em.get("from"), em.get("subject"), preview, em.get("date"), status), tags=(color,))

    # schedule UI update
    root.after(0, ui_update)


def show_email_details(event):
    """
    Show details for the selected email. We use the treeview selection's iid (which is msgid).
    """
    global emails_list, detail_text, model, tokenizer
    sel = email_tree.focus()
    if not sel:
        return

    # iid is string of msgid
    try:
        msgid = int(sel)
    except Exception:
        msgid = None

    selected = None
    for e in emails_list:
        if e.get("id") == msgid:
            selected = e
            break

    if not selected:
        # fallback: try to pick by from+subject
        item = email_tree.item(sel)
        vals = item.get("values", [])
        fromv = vals[0] if len(vals) > 0 else None
        subj = vals[1] if len(vals) > 1 else None
        for e in emails_list:
            if e.get("from") == fromv and e.get("subject") == subj:
                selected = e
                break
    if not selected:
        return

    detail_text.config(state=tk.NORMAL)
    detail_text.delete(1.0, tk.END)

    detail_text.tag_configure("header", font=("Helvetica", 12, "bold"))
    detail_text.tag_configure("body", font=("Helvetica", 10))
    detail_text.tag_configure("phishing", foreground="red", font=("Helvetica", 10, "bold"))
    detail_text.tag_configure("safe", foreground="green", font=("Helvetica", 10, "bold"))

    detail_text.insert(tk.END, "From: ", "header")
    detail_text.insert(tk.END, f"{selected.get('from')}\n", "body")
    detail_text.insert(tk.END, "Subject: ", "header")
    detail_text.insert(tk.END, f"{selected.get('subject')}\n", "body")
    detail_text.insert(tk.END, "Date: ", "header")
    detail_text.insert(tk.END, f"{selected.get('date')}\n\n", "body")

    combined_text = (selected.get("subject") or "") + " " + (selected.get("body") or "")
    if contains_phishing_keywords(combined_text):
        detail_text.insert(tk.END, "Status: ", "header")
        detail_text.insert(tk.END, "Phishing (Keyword)\n\n", "phishing")
    else:
        if model and tokenizer:
            try:
                encoding = tokenizer.encode_plus(
                    combined_text,
                    add_special_tokens=True,
                    max_length=256,
                    return_token_type_ids=False,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, pred = torch.max(outputs.logits, dim=1)
                    pred = int(pred.cpu().item())

                detail_text.insert(tk.END, "Status: ", "header")
                if pred == 1:
                    detail_text.insert(tk.END, "Phishing\n\n", "phishing")
                else:
                    detail_text.insert(tk.END, "Safe\n\n", "safe")
            except Exception:
                detail_text.insert(tk.END, "Status: Error evaluating model\n\n", "body")
        else:
            detail_text.insert(tk.END, "Status: Unknown (model not loaded)\n\n", "body")

    body_text = BeautifulSoup(selected.get("body") or "", "html.parser").get_text()
    detail_text.insert(tk.END, "Body:\n", "header")
    detail_text.insert(tk.END, body_text, "body")
    detail_text.config(state=tk.DISABLED)


# -------------------------
# UI: Main windows
# -------------------------
def open_main_app():
    global root, email_tree, detail_text, model, tokenizer, emails_list

    # ensure model loaded: attempt to load if not already loaded
    if model is None or tokenizer is None:
        # start background thread to train/load
        # we'll show an info box when done via callback
        start_training_and_notify()

    root = tk.Tk()
    root.title("Email Phishing Detector - Gmail Style")
    root.geometry("1200x800")
    root.configure(bg="#F0F0F0")

    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    toolbar = ttk.Frame(main_frame)
    toolbar.pack(fill="x", pady=5)

    ttk.Button(toolbar, text="Scan Emails", command=check_email_threaded).pack(side="left", padx=5)
    ttk.Button(toolbar, text="Scan URL", command=open_vt_window).pack(side="left", padx=5)
    ttk.Button(toolbar, text="Refresh", command=check_email_threaded).pack(side="left", padx=5)

    list_frame = ttk.Frame(main_frame)
    list_frame.pack(fill="both", expand=True)

    email_tree = ttk.Treeview(list_frame,
                              columns=("From", "Subject", "Preview", "Date", "Status"),
                              show="headings", selectmode="browse")
    email_tree.heading("From", text="From", anchor="w")
    email_tree.heading("Subject", text="Subject", anchor="w")
    email_tree.heading("Preview", text="Preview", anchor="w")
    email_tree.heading("Date", text="Date", anchor="w")
    email_tree.heading("Status", text="Status", anchor="w")

    email_tree.column("From", width=200, anchor="w")
    email_tree.column("Subject", width=250, anchor="w")
    email_tree.column("Preview", width=400, anchor="w")
    email_tree.column("Date", width=150, anchor="w")
    email_tree.column("Status", width=100, anchor="w")

    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=email_tree.yview)
    email_tree.configure(yscrollcommand=scrollbar.set)

    email_tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    email_tree.tag_configure("red", foreground="red")
    email_tree.tag_configure("green", foreground="green")

    email_tree.bind("<ButtonRelease-1>", show_email_details)

    detail_frame = ttk.Frame(main_frame)
    detail_frame.pack(fill="both", expand=True, pady=10)

    detail_label = ttk.Label(detail_frame, text="Email Details:", font=("Helvetica", 12, "bold"))
    detail_label.pack(anchor="w")

    detail_text = scrolledtext.ScrolledText(detail_frame, wrap=tk.WORD, font=("Helvetica", 10))
    detail_text.pack(fill="both", expand=True)
    detail_text.config(state=tk.DISABLED)

    # keep a reference so nested functions can access
    root.mainloop()


# -------------------------
# VirusTotal UI
# -------------------------
def open_vt_window():
    vt_window = tk.Toplevel()
    vt_window.title("Scan URL with VirusTotal")
    vt_window.geometry("800x600")

    ttk.Label(vt_window, text="Enter URL to scan:", font=("Helvetica", 12)).pack(pady=10)
    url_entry = ttk.Entry(vt_window, width=70)
    url_entry.pack(pady=5)
    result_text = scrolledtext.ScrolledText(vt_window, wrap=tk.WORD, height=25)
    result_text.pack(pady=10, fill="both", expand=True)

    def append_result(text):
        result_text.insert(tk.END, text + "\n")
        result_text.see(tk.END)

    def submit_url():
        url = url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return

        result_text.delete(1.0, tk.END)
        append_result("Submitting URL...")

        def worker():
            vt_scan_url(url, result_callback=append_result)

        threading.Thread(target=worker, daemon=True).start()

    ttk.Button(vt_window, text="Scan URL", command=submit_url).pack(pady=10)


# -------------------------
# Login UI
# -------------------------
def login():
    global EMAIL, PASSWORD, login_window
    EMAIL = entry_1.get().strip()
    PASSWORD = entry_2.get().strip()

    if not EMAIL or not PASSWORD:
        messagebox.showerror("Error", "Please enter both email and password.")
        return

    # Try to login once to validate credentials (do in a thread so UI doesn't freeze)
    def worker():
        try:
            with imapclient.IMAPClient("imap.gmail.com", ssl=True) as server:
                server.login(EMAIL, PASSWORD)
            # success - switch to main app on main thread
            def open_app():
                try:
                    login_window.destroy()
                except Exception:
                    pass
                open_main_app()
            root_ref = login_window  # closure copy
            login_window.after(0, open_app)

        except Exception as e:
            def show_err():
                messagebox.showerror("Error", f"Login failed: {e}\n\nNote: If you use Gmail, use an App Password, not regular password.")
            login_window.after(0, show_err)

    threading.Thread(target=worker, daemon=True).start()


# Build login window
login_window = tk.Tk()
login_window.title("Login")
login_window.geometry("950x600")
login_window.configure(bg="#FFFFFF")

canvas = tk.Canvas(login_window, bg="#FFFFFF", height=600, width=950, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)

# Assets: try to load relative assets; if missing, ignore images but continue
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / path

# Some assets may not exist; protect with try/except
try:
    image_image_1 = tk.PhotoImage(file=str(relative_to_assets("image_1.png")))
    image_1 = canvas.create_image(241.0, 180.0, image=image_image_1)
except Exception:
    image_image_1 = None

canvas.create_text(316.0, 165.0, anchor="nw", text="Phishing Email Detection", fill="#000000", font=("Inter", 24 * -1))

# Entry backgrounds try load but not required
try:
    entry_image_1 = tk.PhotoImage(file=str(relative_to_assets("entry_1.png")))
    entry_bg_1 = canvas.create_image(450.56, 292.83, image=entry_image_1)
except Exception:
    entry_bg_1 = None

entry_1 = tk.Entry(login_window, bd=0, bg="#E2E6E6", fg="#000716", highlightthickness=0)
entry_1.place(x=350.0, y=274.0, width=201.12, height=35.67)

try:
    entry_image_2 = tk.PhotoImage(file=str(relative_to_assets("entry_2.png")))
    entry_bg_2 = canvas.create_image(450.56, 375.83, image=entry_image_2)
except Exception:
    entry_bg_2 = None

entry_2 = tk.Entry(login_window, bd=0, bg="#E2E6E6", fg="#000716", highlightthickness=0, show="*")
entry_2.place(x=350.0, y=357.0, width=201.12, height=35.67)

canvas.create_text(350.0, 252.0, anchor="nw", text="Your email : ", fill="#000000", font=("Inter", 16 * -1))
canvas.create_text(344.0, 338.0, anchor="nw", text="Your Password :", fill="#000000", font=("Inter", 16 * -1))

# Button: use a regular button if asset missing
try:
    button_image_1 = tk.PhotoImage(file=str(relative_to_assets("button_1.png")))
    button_1 = tk.Button(login_window, image=button_image_1, borderwidth=0, highlightthickness=0, command=login, relief="flat")
    button_1.place(x=392.0, y=419.0, width=74.0, height=26.0)
except Exception:
    button_1 = tk.Button(login_window, text="Login", command=login)
    button_1.place(x=420.0, y=420.0)

login_window.resizable(False, False)
login_window.mainloop()
