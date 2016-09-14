---
layout: page
title: Other Blogs
---

{% for blog in site.data.blogs %}
* {{ blog.name }} &raquo; [ {{ blog.url }} ]({{ blog.url }})
{% endfor %}
