# Practical Data Privacy

Notebooks to accompany the book _Practical Data Privacy: Enhancing Privacy and Security in Data_, O'Reilly, Spring 2023.

There are several notebooks associated with the upcoming O'Reilly video course as well.

You can [read the book on Safari now](https://www.oreilly.com/library/view/practical-data-privacy/9781098129453/). Pre-order is also available.

These notebooks can also be used separately from the book, as a workhop or self-study to learn about practical data privacy methods. The audience is assumed to be a data scientist or data folks with an understanding of probability, math and data. 

### Motivation

The goal of the notebooks and the book is to help data scientists and other technologists learn about practical data privacy. I hope you can use these notebooks and the book to not only learn about data privacy, but also to guide implementation of data privacy in your work!

These notebooks are not meant to replace exploring software or building sustainable, production-ready code; but instead are meant to help guide your learning and thinking around the topics. Please always try to use and support open-source libraries based on the learnings you get from these notebooks / the book.

### Installation

Please utilize the included `requirements.txt` to install your requirements using `pip` (you can also do so in `conda`. The notebooks have *only* been tested with Python 3. 🙌🏻

Unfortunately, some of these libraries have conflicting requirements, so you may need to adapt your libraries and install to use later notebooks after you install the earlier tools. You will also need to install several Rust libraries with Python bindings, which you will need to follow the direct installation information from those software packages.

I recommend using [virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/) or [conda environments](https://conda.io/docs/user-guide/tasks/manage-environments.html). 

To run parts of these notebooks you will also need a running version of Apache Spark. Check [the latest documentation](https://spark.apache.org/downloads.html) to set up for your operating system.


Notebooks
--------

The notebooks follow the order that the ideas are introduced in the book. There are some additional notebooks added for those interested. Please file a pull request if you have an update to a notebook. I will also watch issues to ensure that the notebooks are usable and understandable. Feedback is very welcome!

### Recommended Reading and Challenges

Several notebooks have a recommended reading and additional challenges section. I may update this README with additional reading of interest on this topic. I also recommend that you try out at least one or two challenges, to expand the knowledge you learned and practice using this for new problems.

### Reader Contributions

I'm hoping this book and repository has inspired you to try out new libraries and tools related to data privacy. To encourage yourself and others to share their work, I have a folder here `reader-contributions`. If you try something new out, please consider contributing your notebook! To make it easier for others, please ensure you:

- Write a brief introduction to the concept or library shown in the notebook, including any links for folks to learn more. What will they learn? What does it show?
- Installation requirements
- Your name (if you'd like recognition) and any details should people want to reach out (optional!)
- Guide other readers through the notebook with occasional titles, markdown cells to take someone through the notebook when you cannot be there.
- Recommended Reading or Challenges

Feel free to send over Pull Requests once you've checked the above!

Thank you for your work and contribution, and helping others learn more about privacy!

### Questions?

Questions about getting set up or the content covered in the notebooks or book? Feel free to reach out via email at: katharine (at) kjamistan (dot) com

### Acknowledgements and Contributions

These notebooks wouldn't have been possible without the following contributors and reviewers. Thank you!

- [Damien Desfontaines](https://desfontain.es/serious.html)
- [Morten Dahl](https://github.com/mortendahl)
- [Jason Mancuso](https://github.com/jvmncs)
- [Yann Dupis](https://github.com/yanndupis)
- [Mitchell Lisle](https://github.com/mitchelllisle)

Please note that the dpia-template.docx is downloaded from [the UK data protection authority (ICO)](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/guide-to-accountability-and-governance/accountability-and-governance/data-protection-impact-assessments/) and is meant to be used for educational purposes only.

### Update Log

23.08.2024: Main notebooks and examples for video course added.
20.02.2023: Main notebooks working and added reader-contributions folder.
