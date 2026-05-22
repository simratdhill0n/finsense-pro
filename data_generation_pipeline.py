import pandas as pd
import numpy as np
from faker import Faker
import random
import os
import json

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

fake = Faker('en_CA')

class DataGenerationPipeline:
    def __init__(self, num_samples=10000, anomaly_rate=0.10):
        self.num_samples = num_samples
        self.anomaly_rate = anomaly_rate
        
        self.vendors = [
            "ABC Supplies Ltd", "Maple Tech Solutions", "Toronto Office Depot",
            "Vancouver Logistics Inc", "Ottawa Consulting Group", "Montreal Hardware Co",
            "Calgary Energy Services", "Halifax Shipping Ltd", "Quebec Paper Products"
        ]
        self.vendor_avg_amounts = {v: random.randint(800, 2800) for v in self.vendors}

    def generate_line_items(self, total_amount):
        """Generate realistic line items"""
        num_items = random.randint(2, 6)
        items = []
        remaining = total_amount
        
        for i in range(num_items - 1):
            item_amount = round(np.random.uniform(50, remaining * 0.6), 2)
            items.append({
                "description": random.choice(["Office Supplies", "Software License", "Consulting Services", 
                                            "Hardware Parts", "Shipping Charges", "Maintenance Fee", "Training"]),
                "amount": item_amount
            })
            remaining -= item_amount
        
        items.append({"description": "Total", "amount": round(remaining, 2)})
        return items

    def generate_normal_invoice(self):
        """Generate one highly realistic normal invoice"""
        vendor = random.choice(self.vendors)
        base_amount = self.vendor_avg_amounts[vendor]
        amount = round(np.random.normal(base_amount, base_amount * 0.18), 2)
        
        line_items = self.generate_line_items(amount)
        
        invoice_text = f"""INVOICE
Invoice #: INV-{fake.unique.random_number(digits=6)}
Date: {fake.date_between(start_date='-18m', end_date='today').strftime('%B %d, %Y')}

Bill To:
Simrat Enterprises
123 Maple Avenue, Suite 400
Toronto, ON M5V 2T6

Vendor:
{vendor}
{fake.address().replace('\n', ', ')}

Description                          Amount
----------------------------------------
"""
        for item in line_items[:-1]:
            invoice_text += f"{item['description']:<35} ${item['amount']:,.2f}\n"
        
        invoice_text += f"\n{'-'*40}\n"
        invoice_text += f"Subtotal{'':<29} ${amount:,.2f}\n"
        invoice_text += f"HST (13%){'':<30} ${round(amount*0.13, 2):,.2f}\n"
        invoice_text += f"{'='*40}\n"
        invoice_text += f"Total Due{'':<30} ${amount + round(amount*0.13, 2):,.2f}\n\n"
        invoice_text += "Thank you for your business!"

        invoice = {
            'document_type': 'invoice',
            'invoice_id': fake.unique.random_number(digits=8),
            'vendor_name': vendor,
            'amount': amount,
            'date': fake.date_between(start_date='-18m', end_date='today'),
            'invoice_number': f"INV-{fake.unique.random_number(digits=6)}",
            'tax_rate': 0.13,
            'tax_amount': round(amount * 0.13, 2),
            'line_items': json.dumps(line_items),
            'invoice_text': invoice_text,
            'is_anomaly': False,
            'anomaly_type': 'normal'
        }
        return invoice

    def inject_anomaly(self, invoice):
        """Enhanced with 7 realistic fraud types"""
        anomaly_type = random.choice([
            'amount_spike', 'duplicate', 'fake_vendor', 'unusual_timing',
            'payment_redirection', 'round_amount', 'overbilling'
        ])
        
        if anomaly_type == 'amount_spike':
            invoice['amount'] = round(invoice['amount'] * random.uniform(2.5, 4.5), 2)
            
        elif anomaly_type == 'duplicate':
            invoice['invoice_number'] = invoice['invoice_number'].replace("INV-", "DUP-")
            
        elif anomaly_type == 'fake_vendor':
            invoice['vendor_name'] = fake.company() + " (Suspicious)"
            
        elif anomaly_type == 'unusual_timing':
            invoice['date'] = fake.date_time_between(start_date='-18m', end_date='now').replace(
                hour=random.randint(22, 23), minute=random.randint(0, 59))
            
        elif anomaly_type == 'payment_redirection':
            invoice['invoice_text'] = invoice['invoice_text'].replace(
                "Thank you for your business!", 
                "IMPORTANT: Our bank details have changed. Please pay to new account ending in 7845.\nThank you!")
            
        elif anomaly_type == 'round_amount':
            invoice['amount'] = round(invoice['amount'] / 1000) * 1000
            
        elif anomaly_type == 'overbilling':
            invoice['amount'] = round(invoice['amount'] * random.uniform(1.3, 2.0), 2)
        
        invoice['is_anomaly'] = True
        invoice['anomaly_type'] = anomaly_type
        return invoice

    def run(self):
        print("🚀 Starting Enhanced Realistic Data Generation Pipeline...")
        
        normal_count = int(self.num_samples * (1 - self.anomaly_rate))
        anomaly_count = self.num_samples - normal_count
        
        print(f"Generating {normal_count:,} normal + {anomaly_count:,} fraudulent samples...")
        
        data = [self.generate_normal_invoice() for _ in range(normal_count)]
        
        for _ in range(anomaly_count):
            normal = self.generate_normal_invoice()
            anomalous = self.inject_anomaly(normal)
            data.append(anomalous)
        
        random.shuffle(data)
        df = pd.DataFrame(data)
        
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/invoices_dataset.csv', index=False)
        df.sample(50).to_csv('data/sample_invoices.csv', index=False)
        
        print(f"\n✅ Enhanced dataset generated successfully!")
        print(f"   Total samples : {len(df):,}")
        print(f"   Anomalous     : {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean():.1%})")
        print("\nAnomaly Distribution:")
        print(df[df['is_anomaly']]['anomaly_type'].value_counts())
        
        print("\n💾 Files saved:")
        print("   - data/invoices_dataset.csv (full realistic dataset)")
        print("   - data/sample_invoices.csv (50 samples)")
        
        return df

if __name__ == "__main__":
    pipeline = DataGenerationPipeline(num_samples=10000, anomaly_rate=0.10)
    df = pipeline.run()
    print("\nPreview:")
    print(df[['vendor_name', 'amount', 'is_anomaly', 'anomaly_type']].head())