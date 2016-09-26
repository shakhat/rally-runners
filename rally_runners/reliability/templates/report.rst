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

Etalon
******

{{ item.etalon_table }}

Errors
******

{{ item.errors_table }}

Degradation
***********

{{ item.degradation_table }}

{% endfor %}
