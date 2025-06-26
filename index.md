---
layout: home
title: Blogs, tutorials and research notes
permalink: /
---

## Available tutorials

<ul>
{% assign docs = site.pages | where_exp: "doc", "doc.path contains '_tutorials/'" %}
{% for doc in docs %}
  <li><a href="{{ doc.url | relative_url }}">{{ doc.title }}</a></li>
{% endfor %}
</ul>
