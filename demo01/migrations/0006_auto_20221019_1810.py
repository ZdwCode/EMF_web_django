# Generated by Django 2.0.2 on 2022-10-19 10:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('demo01', '0005_datetest'),
    ]

    operations = [
        migrations.AddField(
            model_name='liquidinfo',
            name='date',
            field=models.DateField(default='2022-01-01'),
        ),
        migrations.AddField(
            model_name='thickinfo',
            name='date',
            field=models.DateField(default='2022-01-01'),
        ),
    ]
