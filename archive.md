---
layout: page
title: Archive
---

## Blog Posts

{% for post in site.posts %}{% if post.completed %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{site.baseurl_}}{{ post.url }})
{% endif %}{% endfor %}