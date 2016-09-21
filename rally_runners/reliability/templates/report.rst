Scenario
~~~~~~~~

.. code-block:: yaml

{{ report.scenario }}

Summary
~~~~~~~

{{ report.summary_table }}


Details
~~~~~~~

{% for item in report.runs %}

Run #{{ loop.index }}
######

.. image:: plot_{{ loop.index }}.svg

Errors
******

{{ item.errors_table }}

Anomalies
*********

{{ item.anomalies_table }}

{% endfor %}
