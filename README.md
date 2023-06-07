# BDA_final
**Big Data Analytics 23 - IBA - MS Data Sciences - Final Group Project**

Project Members:
- Hammad Hadi Khan - 14278
- Hassan Hadi Khan - 12837
		

1. Open command prompt
2. Execute: cd "<local-directory>"
	
	This command will navigate you to the directlry where you want to clone/copy the git files 
3. Execute: git clone https://github.com/HHadiKhan/BDA_final
	
	This will create a copy of repository files in the local directory where you navigated in the above command.
4. Navigate to the cloned folder BDA_final
	
	Execute the cd "" command in step 1 and add "/BDA_final" to enter the BDA_final folder in the directory.
5. Execute: docker-compose build --no-cache
	
	This command will build the image for the services mentioned in the docker-compose file
6. Execute: docker-compose up -d
	
	This command will start the containers/services mentioned in the docker-compose file
7. Execute: docker exec -it streamlitapi sh
	
	By executing this command, you can access the shell for the streamlit container and run commands within it.
8. Execute: streamlit run my_webapp.py
	
	This command will run the my_webapp.py streamlit webapp in the browser.
9. Run in browser if the web app does not load automatically: http://localhost:8502/
	
	If the webapp does not load automatically, then you can type this command in the web browser to manually run this webapp.
