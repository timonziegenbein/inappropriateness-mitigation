# Generated by Django 3.0.3 on 2021-01-17 10:30

import django.contrib.postgres.fields.jsonb
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('study', '0006_auto_20210115_1633'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='annotation',
            name='result',
        ),
        migrations.AddField(
            model_name='annotation',
            name='results',
            field=django.contrib.postgres.fields.jsonb.JSONField(default=dict),
        ),
    ]