import pandas as pd
import numpy as np
from faker import Faker
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

fake = Faker('en_CA')

class DataGenerationPipeline:
    def __init__(self, num_samples=10000, anomaly_rate=0.10):
        self.num_samples = num_samples
        self.anomaly_rate = anomaly_rate
        
        self.document_types = ['invoice', 't4', 'bank_statement', 'expense_report']
        self.vendors = ["ABC Supplies Ltd", "Maple Tech Solutions", "Toronto Office Depot",
                        "Vancouver Logistics Inc", "Ottawa Consulting Group", "Montreal Hardware Co"]

    def generate_document(self):
        """Generate one realistic document of random type"""
        doc_type = random.choice(self.document_types)
        
        if doc_type == 'invoice':
            return self._generate_invoice()
        elif doc_type == 't4':
            return self._generate_t4()
        elif doc_type == 'bank_statement':
            return self._generate_bank_statement()
        else:  # expense_report
            return self._generate_expense_report()

    def _generate_invoice(self):
        vendor = random.choice(self.vendors)
        amount = round(np.random.normal(1500, 600), 2)
        
        text = f"""INVOICE
Invoice #: INV-{fake.unique.random_number(digits=6)}
Date: {fake.date_between(start_date='-18m', end_date='today').strftime('%B %d, %Y')}

Bill To: Simrat Enterprises, Toronto, ON
Vendor: {vendor}

Description                  Amount
-------------------------------
Office Supplies              ${amount:,.2f}
HST (13%)                    ${round(amount*0.13, 2):,.2f}
-------------------------------
Total Due                    ${amount + round(amount*0.13, 2):,.2f}"""

        return {
            'document_type': 'invoice',
            'vendor_name': vendor,
            'amount': amount,
            'date': fake.date_between(start_date='-18m', end_date='today'),
            'invoice_text': text,
            'is_anomaly': False,
            'anomaly_type': 'normal'
        }

    def _generate_t4(self):
        income = round(np.random.normal(65000, 15000), 2)
        return {
            'document_type': 't4',
            'vendor_name': "Canada Revenue Agency",
            'amount': income,
            'date': fake.date_between(start_date='-1y', end_date='today'),
            'invoice_text': f"""T4 Statement of Remuneration Paid\nEmployment Income: ${income:,.2f}\nSIN: {fake.ssn()}""",
            'is_anomaly': False,
            'anomaly_type': 'normal'
        }

    def _generate_bank_statement(self):
        balance = round(np.random.normal(4500, 2000), 2)
        return {
            'document_type': 'bank_statement',
            'vendor_name': "RBC Royal Bank",
            'amount': balance,
            'date': fake.date_between(start_date='-3m', end_date='today'),
            'invoice_text': f"""BANK STATEMENT\nAccount Balance: ${balance:,.2f}\nPeriod: {fake.date_between(start_date='-3m', end_date='today')}""",
            'is_anomaly': False,
            'anomaly_type': 'normal'
        }

    def _generate_expense_report(self):
        amount = round(np.random.normal(850, 400), 2)
        return {
            'document_type': 'expense_report',
            'vendor_name': fake.company(),
            'amount': amount,
            'date': fake.date_between(start_date='-6m', end_date='today'),
            'invoice_text': f"""EXPENSE REPORT\nTotal Expenses: ${amount:,.2f}\nSubmitted by: Employee""",
            'is_anomaly': False,
            'anomaly_type': 'normal'
        }

    def inject_anomaly(self, doc):
        """Apply realistic fraud patterns - fully implemented"""
        if doc['document_type'] == 'invoice':
            anomaly_type = random.choice([
                'amount_spike', 'duplicate', 'fake_vendor', 
                'payment_redirection', 'round_amount', 'overbilling'
            ])
        else:
            anomaly_type = random.choice(['amount_spike', 'fake_vendor', 'unusual_timing'])

        if anomaly_type == 'amount_spike':
            doc['amount'] = round(doc['amount'] * random.uniform(2.5, 4.5), 2)

        elif anomaly_type == 'duplicate' and 'INV-' in doc.get('invoice_text', ''):
            doc['invoice_text'] = doc['invoice_text'].replace("INV-", "DUP-")

        elif anomaly_type == 'fake_vendor':
            doc['vendor_name'] = fake.company() + " (Suspicious)"

        elif anomaly_type == 'unusual_timing':
            doc['date'] = fake.date_time_between(start_date='-18m', end_date='now').replace(
                hour=random.randint(22, 23), minute=random.randint(0, 59))

        elif anomaly_type == 'payment_redirection':
            doc['invoice_text'] += "\n\nIMPORTANT: Our bank details have changed. " \
                                   "Please pay to new account ending in 7845."

        elif anomaly_type == 'round_amount':
            doc['amount'] = round(doc['amount'] / 1000) * 1000

        elif anomaly_type == 'overbilling':
            doc['amount'] = round(doc['amount'] * random.uniform(1.4, 2.2), 2)

        doc['is_anomaly'] = True
        doc['anomaly_type'] = anomaly_type
        return doc

    def run(self):
        print("🚀 Starting Data Generation Pipeline (4 Document Types)...")
        
        normal_count = int(self.num_samples * (1 - self.anomaly_rate))
        anomaly_count = self.num_samples - normal_count
        
        data = []
        print(f"Generating {normal_count:,} normal + {anomaly_count:,} fraudulent samples...")
        
        for _ in range(normal_count):
            doc = self.generate_document()
            data.append(doc)
        
        for _ in range(anomaly_count):
            doc = self.generate_document()
            doc = self.inject_anomaly(doc)
            data.append(doc)
        
        random.shuffle(data)
        df = pd.DataFrame(data)
        
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/invoices_dataset.csv', index=False)
        df.sample(100).to_csv('data/sample_invoices.csv', index=False)
        
        print(f"\n✅ Dataset generated successfully!")
        print(f"   Total samples : {len(df):,}")
        print(f"   Anomalous     : {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean():.1%})")
        print("\nDocument Type Distribution:")
        print(df['document_type'].value_counts())
        print("\nAnomaly Distribution:")
        print(df[df['is_anomaly']]['anomaly_type'].value_counts())
        
        print("\n💾 Files saved:")
        print("   - data/invoices_dataset.csv")
        print("   - data/sample_invoices.csv")
        
        return df

if __name__ == "__main__":
    pipeline = DataGenerationPipeline(num_samples=10000, anomaly_rate=0.10)
    df = pipeline.run()
    print("\nFirst 5 rows:")
    print(df[['document_type', 'vendor_name', 'amount', 'is_anomaly', 'anomaly_type']].head())