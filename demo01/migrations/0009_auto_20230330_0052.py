# Generated by Django 2.0.2 on 2023-03-29 16:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('demo01', '0008_lastid'),
    ]

    operations = [
        migrations.CreateModel(
            name='LastId2',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('lastid_1', models.IntegerField()),
                ('lastid_2', models.IntegerField()),
                ('lastid_3', models.IntegerField()),
                ('lastid_4', models.IntegerField()),
            ],
        ),
        migrations.AlterField(
            model_name='thickinfo',
            name='date',
            field=models.DateTimeField(default='2022-01-01 00:00:00'),
        ),
    ]
