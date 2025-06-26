---
layout: single
title: 💬 Welcome to my blog page!
permalink: /
---

This is a collection of tutorials and blog posts about my interests and research topics.

<a id="blogs"></a>

## Blog posts

<ul>
{% assign docs = site.blogs | sort: "nav_order" %}
{% for doc in docs %}
  <li><a href="{{ doc.url | relative_url }}">{{ doc.title }}</a></li>
{% endfor %}
</ul>
