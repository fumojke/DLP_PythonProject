import React, { useEffect, useState } from "react";
import axios from "axios";

function App() {
    const [emails, setEmails] = useState([]);
    const [selectedEmail, setSelectedEmail] = useState(null);

    useEffect(() => {
        axios.get("http://localhost:8000/emails")
            .then(response => setEmails(response.data))
            .catch(error => console.error("Помилка загрузки листів:", error));
    }, []);

    const loadEmail = (id) => {
        axios.get(`http://localhost:8000/emails/${id}`)
            .then(response => setSelectedEmail(response.data))
            .catch(error => console.error("Помилка загрузки листа:", error));
    };

    return (
        <div style={{ display: "flex", padding: "20px" }}>
            <div style={{ width: "40%", borderRight: "1px solid #ccc", padding: "10px" }}>
                <h2>📩 E-mail list</h2>
                <ul>
                    {emails.map(email => (
                        <li key={email.id} onClick={() => loadEmail(email.id)}
                            style={{
                                cursor: "pointer",
                                padding: "8px",
                                background: email.is_sensitive ? "#ffcccc" : "#fff"
                            }}>
                            <strong>{email.subject}</strong> ({email.from})
                        </li>
                    ))}
                </ul>
            </div>
            <div style={{ width: "60%", padding: "10px" }}>
                {selectedEmail ? (
                    <div>
                        <h2>📜 E-mail details</h2>
                        <p><strong>From:</strong> {selectedEmail.from}</p>
                        <p><strong>To:</strong> {selectedEmail.to}</p>
                        <p><strong>Subject:</strong> {selectedEmail.subject}</p>
                        <p><strong>Body:</strong> {selectedEmail.text}</p>
                        {selectedEmail.is_sensitive && <p style={{ color: "red" }}>⚠️ Містить конфіденційну інформацію!</p>}
                    </div>
                ) : (
                    <p>Select an e-mail to view</p>
                )}
            </div>
        </div>
    );
}

export default App;
