---
layout: page
title: Contributors
---

{% for author in site.data.authors %}

<div style="display:block;overflow:hidden;margin-top:30px">
	<img src="{{ author[1].pic }}" height=150 style="float:left;margin-right:30px;padding-top:10px;">
	<h2 id="{{ author[0] }}" style="margin-top:0px"> {{ author[1].name }} </h2>
	<p style="display:block;overflow:hidden"> {{ author[1].info }} </p>
</div>

{% endfor %}
